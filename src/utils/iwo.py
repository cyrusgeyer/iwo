import torch
import numpy as np


def log_to_base(tensor: torch.Tensor, base: float) -> torch.Tensor:
    """
    Compute the logarithm of all elements in the input tensor to the specified base.

    Parameters:
    tensor (torch.Tensor): The input tensor.
    base (float): The logarithmic base to use.

    Returns:
    torch.Tensor: A tensor where each element is the logarithm to the
    specified base of the corresponding element in the input tensor.
    """
    # Convert the base to a tensor and ensure it is the same dtype as the input tensor
    base_tensor = torch.tensor([base], dtype=tensor.dtype, device=tensor.device)

    # Calculate the natural logarithm of the tensor and the logarithm of the base
    log_x = torch.log(tensor)
    log_base = torch.log(base_tensor)

    # Apply the change of base formula
    return log_x / log_base


def complete_orthonormal_basis(Q: torch.Tensor) -> torch.Tensor:
    """
    Complete the orthonormal basis given k-1 orthonormal vectors of k dimensions.

    Parameters:
    Q (torch.Tensor): A tensor of shape (k-1, k) containing k-1 orthonormal vectors.

    Returns:
    Tensor: The k-th vector to complete the orthonormal basis.
    """
    # get the dimension of the vectors
    k = Q.size(1)

    # Create a random vector w
    w = torch.randn(k).to(Q.device).to(Q.dtype)

    # Ensure w is not in the span of vectors in Q by checking
    # if the orthogonalization of w results in a zero vector
    while torch.allclose(
        complete_orthonormal_basis_helper(Q, w),
        torch.zeros(k).to(Q.device).to(Q.dtype),
        atol=1e-6,
    ):
        w = torch.randn(k)  # pick a new random vector w and try again

    qk = complete_orthonormal_basis_helper(Q, w)
    return (qk / qk.norm()).unsqueeze(0)  # normalize the vector qk before returning it


def complete_orthonormal_basis_helper(Q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Helper function to orthogonalize w with respect to the vectors in Q.
    """
    for i in range(Q.size(0)):
        qi = Q[i]
        projection = (w.dot(qi) / qi.dot(qi)) * qi
        w = w - projection
    return w


def get_basis(W_list: list, new_dtype=torch.float64) -> torch.Tensor:
    """generate basis matrix B of dimensions L x L where L is the latent space dimension.
    B is orthonormal, the columns are the basis vectors ordered from most important to least important.

    Args:
        W_list (list): learned matrices W_L
        new_dtype (_type_, optional): . Defaults to torch.float64.

    Returns:
        torch.Tensor: basis matrix B
    """
    old_dtype = W_list[0].dtype
    device = W_list[0].device

    if old_dtype != new_dtype:
        W_list = [m.to(new_dtype) for m in W_list]

    prod_W = torch.eye(W_list[0].shape[1]).to(device).to(new_dtype)
    B = []

    # get bL to b2
    for W in W_list:
        Q, _ = torch.linalg.qr(W.t(), mode="complete")
        q = Q[-1:, :]
        b = prod_W @ q.t()
        B.append(b / torch.linalg.norm(b))
        if W.shape[0] != 1:
            prod_W = torch.linalg.lstsq(W.t(), prod_W.t()).solution.t()

    # get b1
    b = prod_W @ W.t()
    B.append(b / torch.linalg.norm(b))

    # get i.o.o. basis
    B = torch.concat(list(reversed(B)), axis=1)

    B = B.to(old_dtype)

    return B


def get_iwo(importance_list: list, Bs: list) -> list:
    """Calculate IWO.

    Args:
        importance_list (list): list of floats holding the relative importance of the basis vectors in B
        Bs (list): _description_

    Returns:
        list: _description_
    """

    num_factors = len(Bs)
    iwo_list = []

    for i in range(num_factors):
        for j in range(i + 1, len(Bs)):
            iwo = (
                1
                - (
                    importance_list[j].unsqueeze(1) ** (1 / 2)
                    * (torch.matmul(Bs[j], Bs[i].transpose(1, 0)) ** 2)
                    * importance_list[i].unsqueeze(0) ** (1 / 2)
                ).sum()
            )
            iwo_list.append(iwo)
    return iwo_list


def get_importance(baseline: float, scores: list, num_dim=None) -> list:
    """Computes the importance as deduced from the loss differences between the NN-heads.

    Args:
        baseline (float): The score which corresponds to random guessing for this task.
        scores (list): The scores that were allocated to the the basis vectors (loss differences between NN-heads)
        keep_small_scores (bool, optional): Whether to keep very small scores or neglect them. Defaults to True.
        num_dim (int, optional): The number of dimensions in the latent space. If not specified, it's assumed to be the length of the list scores.

    Returns:
        list: importance as deduced from the loss differences between the NN-heads
    """

    if num_dim is None:
        num_dim = len(scores) + 1

    device = scores.device

    min_score = baseline
    diff_scores = []
    for j in reversed(range(len(scores))):
        diff = min_score - scores[j]
        if diff > 0:
            diff_scores.append(diff)
            min_score = scores[j]
        else:
            diff_scores.append(0)

    diff_scores = torch.Tensor(diff_scores).to(device)

    if diff_scores.sum() > 0:
        importance = diff_scores / baseline

        if importance.sum().item() < 1:
            # distribute the remaining importance equally
            importance = importance + (1 - importance.sum()) / len(importance)
        importance = importance.to(device)
    else:
        importance = torch.ones_like(diff_scores) / len(diff_scores)
    iwr = (
        (1 - (-importance * log_to_base(importance + 0.0001, num_dim)).sum(0))
        .sum()
        .item()
    )

    return importance, iwr


def calculate_iwo(Bs: list, all_scores: list, baselines: list, num_dim=None):
    """
    Compute the IWO and IWR scores for a given set of Bs, scores and baselines.
    Parameters:
    Bs (list): A list of matrices, each spanning an i.o.o. basis (one for each factor)
    all_scores (list): A nested list of scores associated to the basis vectors in each Q (acquired either through training, validation or testing).
    baselines (list): A list of baselines associated to the factors. (e.g. 5/6 for 6 classes of equal frequency)
    num_dim (int): The number of dimensions in the latent space. For simple cases this equal to the number of basis vectors in each Q.

    returns:
    iwo_list (list): A list of IWO scores.
    iwr_list (list): A list of IWR scores.
    importance (list): A list of importance scores.
    """

    iwr_list = []
    importance_list = []  # basis vector importance

    for i, scores in enumerate(all_scores):
        importance, iwr = get_importance(baselines[i], scores, num_dim)
        importance_list.append(importance)
        iwr_list.append(iwr)

    iwo_list = get_iwo(importance_list, Bs)

    mean_iwr = np.mean(iwr_list)
    mean_iwo = np.mean(iwo_list)

    return iwo_list, iwr_list, mean_iwo, mean_iwr, importance_list


def get_Qs(W_list):
    Q_list = []
    for W in W_list:
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
