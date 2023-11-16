import torch
import numpy as np


def log_to_base(tensor, base):
    """
    Compute the logarithm of all elements in the input tensor to the specified base.

    Parameters:
    tensor (torch.Tensor): The input tensor.
    base (float): The logarithmic base to use.

    Returns:
    torch.Tensor: A tensor where each element is the logarithm to the specified base of the corresponding element in the input tensor.
    """
    # Convert the base to a tensor and ensure it is the same dtype as the input tensor
    base_tensor = torch.tensor([base], dtype=tensor.dtype, device=tensor.device)

    # Calculate the natural logarithm of the tensor and the logarithm of the base
    log_x = torch.log(tensor)
    log_base = torch.log(base_tensor)

    # Apply the change of base formula
    return log_x / log_base


def complete_orthonormal_basis(V):
    """
    Complete the orthonormal basis given k-1 orthonormal vectors of k dimensions.

    Parameters:
    V (Tensor): A tensor of shape (k-1, k) containing k-1 orthonormal vectors.

    Returns:
    Tensor: The k-th vector to complete the orthonormal basis.
    """
    k = V.size(1)  # get the dimension of the vectors
    # Create a random vector w
    w = torch.randn(k).to(V.device)
    # Ensure w is not in the span of vectors in V by checking if the orthogonalization of w results in a zero vector
    while torch.allclose(
        complete_orthonormal_basis_helper(V, w), torch.zeros(k).to(V.device), atol=1e-6
    ):
        w = torch.randn(k)  # pick a new random vector w and try again

    vk = complete_orthonormal_basis_helper(V, w)
    return (vk / vk.norm()).unsqueeze(0)  # normalize the vector vk before returning it


def complete_orthonormal_basis_helper(V, w):
    """
    Helper function to orthogonalize w with respect to the vectors in V.
    """
    for i in range(V.size(0)):
        vi = V[i]
        projection = (w.dot(vi) / vi.dot(vi)) * vi
        w = w - projection
    return w


def get_basis(m_list):
    num_m = len(m_list)
    W = [propagate_forward(m_list[:-1], m_list[-1])]
    for i in reversed(range(num_m - 1)):
        Q = get_Q(m_list[i + 1])
        orth_row = complete_orthonormal_basis(Q)
        b_new = torch.matmul(orth_row, m_list[i])
        W.append(propagate_forward(m_list[:i], b_new))
    complete_Q = get_Q(torch.concat(W))
    return complete_Q


def propagate_forward(w_list, x):
    for w in reversed(w_list):
        x = torch.matmul(x, w)
    return x


def calculate_iwo(Qs, scores, baselines, keep_small_scores=True, num_dim=10):
    """
    Compute the IWO and IWR scores for a given set of Qs, scores and baselines.
    Parameters:
    Qs (list): A list of Q matrices spanning a basis (one for each factor)
    scores (list): A nested list of scores associated to the basis vectors in each Q (acquired either through training, validation or testing).
    baselines (list): A list of baselines associated to the factors. (e.g. 5/6 for 6 classes of equal frequency)
    keep_small_scores (bool): Whether to keep small scores or not. Default is True.
    num_dim (int): The number of dimensions in the latent space. For simple cases this equal to the number of basis vectors in each Q.

    returns:
    iwo_list (list): A list of IWO scores.
    iwr_list (list): A list of IWR scores.
    importance (list): A list of importance scores.
    """

    iwr_list = []
    abs_min_score = []  # minimum score for each factor
    importance = []  # basis vector importance

    for i in range(len(scores)):
        min_score = baselines[i]
        score = []
        ll = baselines[i]
        for j in reversed(range(len(scores[i]))):
            s = scores[i][j]
            _score = ll - s
            if keep_small_scores:
                score.append(max(_score, 0))
            else:
                score.append(_score if (min_score > s and _score > 0.1) else 0)
            min_score = min(min_score, s)
            ll = min_score
        score = torch.Tensor(score).to(scores[i].device)
        if score.sum() > 0:
            if keep_small_scores:
                abs_score = score / baselines[i]
                abs_score = abs_score + (1 - abs_score.sum()) / len(
                    abs_score
                )  # distribute the remaining importance equally
                importance.append(abs_score.to(score.device))
            else:
                importance.append(score / score.sum())
        else:
            importance.append(torch.ones_like(score) * 1 / len(score))
        iwr_list.append(
            (
                1
                - (-importance[-1] * log_to_base(importance[-1] + 0.0001, num_dim)).sum(
                    0
                )
            )
            .sum()
            .item()
        )  # compute IWR
        abs_min_score.append(scores[i].min())

    # Compute relative informativeness about the factos for weighing of IWR later on.
    scr = (torch.tensor(baselines) - torch.tensor(abs_min_score)) / torch.tensor(
        baselines
    )

    iwo_list = []

    for i in range(len(Qs)):
        for j in range(i + 1, len(Qs)):
            iwo = (
                importance[j].unsqueeze(1) ** (1 / 2)
                * (torch.matmul(Qs[j], Qs[i].transpose(1, 0)) ** 2)
                * importance[i].unsqueeze(0) ** (1 / 2)
            ).sum()
            iwo_list.append(-iwo.item())

    mw_iwo = np.mean(iwo_list)
    # mean iwo
    mw_iwr = np.mean(iwr_list)
    # mean iwr

    return iwo_list, iwr_list, mw_iwo, mw_iwr, importance


def get_Qs(m_list):
    Q_list = []
    for W in m_list:
        Q_list.append(get_Q(W))
    return Q_list


def get_Q(W):
    W.to(torch.float64)
    Q, _ = torch.linalg.qr(W.transpose(0, 1))
    return Q.transpose(1, 0)


def allocate_iwo(all_outs, cfg):
    iwo_dict = {}
    iwo_dict["iwo"] = []
    iwo_dict["mean_iwo"] = []
    iwo_dict["rank"] = []
    iwo_dict["mean_rank"] = []
    iwo_dict["importance"] = []

    baselines = []
    for out in all_outs:
        baselines.append(out["baseline"])
    for i in range(len(all_outs[0]["scores"])):
        Qs = []
        scores = []
        for out in all_outs:
            Qs.append(out["Qs"][i])
            scores.append(out["scores"][i])
        iwo_list, iwr_list, mw_iwo, mw_iwr, importance = calculate_iwo(
            Qs, scores, baselines, (not cfg.representation.name == "synthetic")
        )
        iwo_dict["iwo"].append(iwo_list)
        iwo_dict["rank"].append(iwr_list)
        iwo_dict["mean_iwo"].append(mw_iwo)
        iwo_dict["mean_rank"].append(mw_iwr)
        iwo_dict["importance"].append(importance)

    return iwo_dict
