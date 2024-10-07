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

    if W_list[0].shape[0] != W_list[0].shape[1]:
        W_prod = torch.eye(W_list[0].shape[1], device=device, dtype=new_dtype)
    else:
        W_prod = W_list[0]
        W_list.pop(0)

    for i, W in enumerate(W_list):
        W_prod = W @ W_prod
        if i == 0:
            Qr, _ = torch.linalg.qr(W_prod.t(), mode="complete")
            b_list = [Qr[:, -1:]]
        else:
            T = torch.concat([W_prod.t()] + b_list, axis=1)
            Qr, _ = torch.linalg.qr(T, mode="complete")
            b_list.append(Qr[:, -1:])
    T = torch.concat(b_list, axis=1)
    Qr, _ = torch.linalg.qr(T, mode="complete")
    b_list.append(Qr[:, -1:])
    B_flipped = torch.concat(b_list, axis=1)
    B = torch.flip(B_flipped, dims=[1])
    B = B.to(old_dtype)
    return B


def get_iwo(importance_list: list, Bs: list, per_factor_performance: list) -> list:
    """Calculate IWO.

    Args:
        importance_list (list): list of floats holding the relative importance of the basis vectors in B
        Bs (list): _description_

    Returns:
        list: _description_
    """

    num_factors = len(Bs)
    iwo_list = []
    weights = []

    use_weights = per_factor_performance is not None

    for i in range(num_factors):
        for j in range(i + 1, len(Bs)):
            iwo = (
                1
                - (
                    importance_list[j].unsqueeze(1) ** (1 / 2)
                    * (torch.matmul(Bs[j].t(), Bs[i]) ** 2)
                    * importance_list[i].unsqueeze(0) ** (1 / 2)
                ).sum()
            )
            iwo_list.append(iwo)
            if use_weights:
                weights.append(per_factor_performance[i] * per_factor_performance[j])

    if use_weights:
        mean_iwo = weighted_average(iwo_list, weights)
    else:
        mean_iwo = torch.Tensor(iwo_list).mean().item()

    return iwo_list, mean_iwo


def weighted_average(iwo_list, weights):

    weights = torch.Tensor(weights)
    weights = weights / weights.sum()

    return (torch.Tensor(iwo_list) * weights).sum().item()


def get_iwo_entropy_based(importance_list, Bs, per_factor_performance):

    num_factors = len(Bs)
    iwo_list = []

    use_weights = per_factor_performance is not None

    for i in range(num_factors):
        ps = []
        for j in range(len(Bs)):
            if i == j:
                ps.append(importance_list[i])  #
                pass
            else:
                ps.append(
                    (
                        importance_list[i].unsqueeze(1) ** (1 / 2)
                        * torch.matmul(Bs[i].t(), Bs[j]) ** 2
                        * importance_list[j].unsqueeze(0) ** (1 / 2)
                    ).sum(1)
                )
        norm = torch.stack(ps).sum(0, keepdim=True)
        P = torch.stack(ps) / (norm + 0.0001)
        iwo = (
            1
            - (
                (-P * log_to_base(P + 0.00001, num_factors)).sum(0) * importance_list[i]
            ).sum()
        )  #
        iwo = iwo.item()
        iwo_list.append(iwo)

    if use_weights:
        mean_iwo = weighted_average(iwo_list, per_factor_performance)
    else:
        mean_iwo = torch.Tensor(iwo_list).mean().item()

    return iwo_list, mean_iwo


def get_importance(
    baseline: float, scores: list, num_dim=None, importance_mode="fill_up", power=2
) -> list:
    """Computes the importance as deduced from the loss differences between the NN-heads.

    Args:
        baseline (float): The score which corresponds to random guessing for this task.
        scores (list): The scores that were allocated to the the basis vectors (loss differences between NN-heads)
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
        diff = min_score - scores[j].cpu()
        if diff > 0:
            diff_scores.append(diff)
            min_score = scores[j].cpu()
        else:
            diff_scores.append(0)

    diff_scores = torch.Tensor(diff_scores).to(device)

    if diff_scores.sum() > 0:
        if importance_mode == "fill_up":
            importance = diff_scores / baseline
            if importance.sum().item() < 1:
                # distribute the remaining importance equally
                importance = importance + (1 - importance.sum()) / len(importance)
        elif importance_mode == "normalize":
            importance = diff_scores / diff_scores.sum()
        elif importance_mode == "transform_and_normalize":
            importance = diff_scores / baseline
            importance = transform_and_normalize(importance.to(device), power)
        elif importance_mode == "raw":
            importance = diff_scores / baseline
    else:
        importance = torch.ones_like(diff_scores) / len(diff_scores)
    iwr = (
        (1 - (-importance * log_to_base(importance + 0.0001, num_dim)).sum(0))
        .sum()
        .item()
    )

    return importance, iwr, min_score / baseline


def transform_and_normalize(tensor, power=2):
    """
    Transforms a PyTorch tensor such that small values become smaller and large values
    become larger, and then normalizes the tensor so that it sums to one.

    Parameters:
        tensor (torch.Tensor): Input tensor that sums to one.
        power (float): The power to which each element is raised. A power > 1
                       will enhance the difference between small and large values.

    Returns:
        torch.Tensor: A new tensor that sums to one with exaggerated small and large values.
    """
    # Apply the power transformation
    transformed_tensor = tensor**power

    # Normalize the tensor to sum to 1
    new_tensor = transformed_tensor / torch.sum(transformed_tensor)

    return new_tensor


def calculate_iwo(
    Bs: list,
    all_scores: list,
    baselines: list,
    num_dim: int = None,
    importance_mode: str = "fill_up",
    iwo_mode: str = "classic",
    normalize_over_all_factors: bool = False,
    performance_weighted_average: bool = False,
    power: float = 2.0,
):
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
    importance_list = []
    min_score_list = []

    for i, scores in enumerate(all_scores):
        importance, iwr, min_score = get_importance(
            baselines[i], scores, num_dim, importance_mode, power
        )
        importance_list.append(importance)
        iwr_list.append(iwr)
        min_score_list.append(min_score)

    mean_iwr = np.average(iwr_list)

    if normalize_over_all_factors:
        total_importance = 0
        for importance in importance_list:
            total_importance += torch.sum(importance)
        importance_list = [imp / total_importance for imp in importance_list]

    if performance_weighted_average:
        min_score_list = np.array(min_score_list)
        per_factor_performance = 1 - min_score_list
        per_factor_performance[per_factor_performance < 0] = 0
        if per_factor_performance.sum() > 0:
            per_factor_performance = (
                per_factor_performance / per_factor_performance.sum()
            )
        else:
            per_factor_performance = np.ones(len(per_factor_performance)) / len(
                per_factor_performance
            )
    else:
        per_factor_performance = None

    if iwo_mode == "classic":
        iwo_list, mean_iwo = get_iwo(importance_list, Bs, per_factor_performance)
    if iwo_mode == "entropy":
        iwo_list, mean_iwo = get_iwo_entropy_based(
            importance_list, Bs, per_factor_performance
        )

    return (
        iwo_list,
        iwr_list,
        mean_iwo,
        mean_iwr,
        importance_list,
        1 - np.mean(min_score_list),
    )
