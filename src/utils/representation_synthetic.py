import numpy as np
import torch
from torch.utils.data import Dataset
import math

# Function which genrate the factors first and then the latents from them:


def generate_generative_factors(n=1000, d=5, factor_sizes=None):
    if factor_sizes is None:
        factor_sizes = [0, 0.25, 0.5, 0.75, 1.0]
    return np.random.choice(factor_sizes, size=(n, d))


def generate_noisy_data(n, d, sigma=0.001, factor_sizes=None):
    gen_factors = generate_generative_factors(n, d, factor_sizes)
    latent_reps = gen_factors + np.random.normal(size=(n, d), scale=sigma)
    return [gen_factors, latent_reps]


def generate_permuted_data(n, d, factor_sizes=None):
    gen_factors = generate_generative_factors(n, d, factor_sizes)
    latent_reps = gen_factors[:, np.random.permutation(d)]
    return [gen_factors, latent_reps]


def get_random_orthonormal_base(d):
    A = np.random.randn(d, d)
    Q, _ = np.linalg.qr(A)
    return Q


def generate_orthogonal_data(n, d, factor_sizes=None):
    gen_factors = generate_generative_factors(n, d, factor_sizes)
    Q = get_random_orthonormal_base(d)
    latent_reps = np.matmul(gen_factors, Q)
    return [gen_factors, latent_reps]


# Functions which genrate the latents first and then the factors from them:


def generate_latent_variables(n, l, sigma):
    return np.random.normal(size=(n, l), scale=sigma)


def normalize_and_quantize_gen_factors(gen_factors, percentile_to_clip=1, bins=40):
    _min = np.percentile(gen_factors, percentile_to_clip)
    _max = np.percentile(gen_factors, 100 - percentile_to_clip)

    gen_factors = np.clip(gen_factors, _min, _max)

    normalized_gen_factors = (gen_factors - _min) / (_max - _min)

    return np.round(normalized_gen_factors * bins) / bins


def generate_data(
    N=1000,
    L=10,
    K=5,
    sigma=1.0,
    rank=2,
    random_orthogonal_projection=False,
    func="quadratic",
):
    latent_reps = generate_latent_variables(N, L, sigma)
    assert L % K == 0
    assert L >= K
    assert rank <= L

    latent_reps_repeated = np.hstack([latent_reps, latent_reps, latent_reps])

    i = int(rank / 2 + 0.5)  # This rounds rank/2 up.
    j = i - (rank % 2)  # This is i if rank is even and i-1 if rank is odd.

    gen_list = []
    for l in range(L, 2 * L, L // K):
        c_span = latent_reps_repeated[:, l - j : l + i]

        if func == "linear":
            gen_list.append(
                np.mean(c_span, -1)
            )  # Note that a linear recombination leads to the data living in a single dimension.
        elif func == "quadratic":
            gen_list.append((np.sqrt(np.mean(c_span**2, -1))))
        elif func == "trig":
            gen_list.append((np.mean(np.cos(c_span), -1)))
        else:
            raise ValueError(
                f"Invalid function type: {func}. Expected 'linear', 'quadratic', or 'trig'."
            )

    gen_factors = np.vstack(gen_list).transpose()
    gen_factors = normalize_and_quantize_gen_factors(gen_factors)

    if random_orthogonal_projection:
        Q = get_random_orthonormal_base(L)
        latent_reps = np.matmul(latent_reps, Q)

    return [gen_factors, latent_reps]


# Define a custom dataset
class SyntheticDataset(Dataset):
    def __init__(self, data, targets, factor_discrete):
        self.targets = torch.FloatTensor(targets)
        self.data = torch.FloatTensor(data)
        self._factor_discrete = factor_discrete

        K = self.targets.shape[-1]
        self._factor_sizes = [len(np.unique(self.targets[:, k])) for k in range(K)]
        self.normalized_targets = (
            self.targets - self.targets.min()
        ) / self.targets.max()
        self._factor_names = [str(i) for i in range(K)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.normalized_targets[index]
        return x, y


def make_datasets(cfg, split=None):
    N = cfg.N
    K = cfg.K
    if cfg.function == "noisy":
        sigma = cfg.noise_sigma
        targets, data = generate_noisy_data(N, K, sigma)
        _factor_discrete = [True for k in range(K)]
    elif cfg.function == "permuted":
        targets, data = generate_permuted_data(N, K)
        _factor_discrete = [True for k in range(K)]
    else:
        L = cfg.L
        sigma = cfg.data_sigma
        rank = cfg.rank
        random_orthogonal_projection = cfg.orth_proj
        func = cfg.function
        targets, data = generate_data(
            N, L, K, sigma, rank, random_orthogonal_projection, func
        )
        _factor_discrete = [False for k in range(K)]

    if split == None:
        split = [0.7, 0.15]

    train_end_idx = int(split[0] * N)
    val_end_idx = train_end_idx + int(split[1] * N)

    splits = [train_end_idx, val_end_idx]

    train_dataset = SyntheticDataset(
        data[: splits[0]], targets[: splits[0]], _factor_discrete
    )
    val_dataset = SyntheticDataset(
        data[splits[0] : splits[1]], targets[splits[0] : splits[1]], _factor_discrete
    )
    test_dataset = SyntheticDataset(
        data[splits[1] :], targets[splits[1] :], _factor_discrete
    )

    return train_dataset, val_dataset, test_dataset
