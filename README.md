# Measuring Orthogonality in Representations of Generative Models

## Introduction
This repository hosts the code and resources for the research paper "Measuring Orthogonality in Representations of Generative Models". It's designed to be used in conjunction with the [disentanglement_lib](https://github.com/google-research/disentanglement_lib) repository, but also includes synthetic experiments that can be run independently.

## Features
- Integration with `disentanglement_lib` for comprehensive disentanglement studies.
- Support for datasets such as DSprites, cars3D, and SmallNorb.
- Synthetic experiments for immediate execution.
- Configuration management via [Hydra](https://hydra.cc)
- Slurm cluster support with Hydra's Submitit plugin.

## Getting Started

### Prerequisites
- Python 3.11 (recommended)
- Virtual environment (recommended)

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/{anonymous}/iwo.git
   ```
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate
   ```
3. Install the requirements:
   ```
   pip install lightning numpy hydra-core 'wandb>=0.12.10' 
   ```

### Usage
#### Basic Usage
Run synthetic experiments with the following command:
```
python main.py --config-dir conf --config-name synthetic
```

#### Using with disentanglement_lib
To use this repository with `disentanglement_lib`, you need to first extract the representations and factors as `.npy` files.
A supporting script follows soon. 

#### Cluster Deployment
This repository supports experiments on a Slurm-type cluster using the Submitit plugin for Hydra. Configuration details can be found in `conf/hydra/submitit_gpu`.

## What to expect
### Synthetic experiments

![joint](./assets/loss_plot.png?raw=true)

In the above figure we depict the loss of neural network heads at different projection steps for a single run of synthetic experiment 3 of our paper ($L = 10$, $K = 5$ and $R_j = 5$). $\mathcal{L}_{l}$ for $l \geq 6$ are omitted , as they are almost zero (similar to $\mathcal{L}_5$). 
Each generative factor is analysed using GCA, which means for each generative factor we train an LNN spine with 9 matrices $W_9 \in \mathbb{R}^{9 \times 10}$ ... $W_1 \in \mathbb{R}^{1 \times 2}$ and 10 NN-heads acting on the projections. 
Because of the symmetry of the synthetic experiments, all five generative factors are similarly encoded in the latent space. In the above figure we are therefore depicting the mean and standard deviation over the generative factors. 
An entire pass through each LNN projects the representation to the most informative dimension for the respective generative factor. When recovering the generative factor from that projection, we incur a loss of $\mathcal{L}_1$. The fraction $\mathcal{L}_1 / \mathcal{L}_0$ tells us that this is  $\approx 20\%$  better than naive guessing (assuming the expectation value) of the factor. We see that we can almost perfectly recover the generative factors from projections to 5 and more dimensions. This is expected as the experiment is setup with $R_j = 5$. We see that each subsequently removed dimension increases the loss by $\approx 20\%$. It follows that $\Delta \mathcal{L}_l / \mathcal{L}_0  \approx 0.2$, i.e. $\omega_l \approx 0.2$ for $1 \leq l \leq 5$. The right hand side of Figure \ref{fig:loss} depicts the relative validation loss of the NN-heads during training. The relative loss at the 72nd iteration corresponds to the relative loss depicted in the left plot. }


## Contributing
Contributions to improve the repository or the research are welcome. Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests. [coming soon]

## License
This project is licensed under the MIT License- see the `LICENSE` file for details.

## Acknowledgments
- We thankthe creators of [DCI-ES](https://github.com/andreinicolicioiu/DCI-ES) for the inspiration that their code gave us. 
- [More will follow]

## Contact
For any queries or further discussion, feel free to contact us at [anonymous].

