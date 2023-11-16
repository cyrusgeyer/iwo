# Beyond Disentanglement: On the Orthogonality of Learned Representations

## Introduction
This repository hosts the code and resources for the research paper "Beyond Disentanglement: On the Orthogonality of Learned Representations". It's designed to be used in conjunction with the [disentanglement_lib](https://github.com/google-research/disentanglement_lib) repository, but also includes synthetic experiments that can be run independently.

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

## Contributing
Contributions to improve the repository or the research are welcome. Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests. [coming soon]

## License
This project is licensed under the MIT License- see the `LICENSE` file for details.

## Acknowledgments
- We thankthe creators of [DCI-ES](https://github.com/andreinicolicioiu/DCI-ES) for the inspiration that their code gave us. 
- [More will follow]

## Contact
For any queries or further discussion, feel free to contact us at [anonymous].

