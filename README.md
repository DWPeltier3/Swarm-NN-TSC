# Swarm Characteristics Classification Using Neural Networks

## Purpose

This repository hosts the source code used in three related research studies focused on classifying swarm characteristics using neural network-based time series classification (TSC). These works explore deterministic, robust, and Bayesian neural network approaches applied to adversarial swarm-on-swarm scenarios:

1. Peltier et al., *Swarm characteristics classification using neural networks*, IEEE TAES, 2025 [[1]](#citations)
2. Peltier et al., *Swarm characteristics classification using robust neural networks with optimized controllable inputs*, arXiv, 2025 [[2]](#citations)
3. Peltier et al., *Swarm characteristics classification uncertainty using Bayesian neural networks*, manuscript pending submission, 2025 [[3]](#citations)

The goal of this project is to provide a transparent, reproducible, and extensible implementation of the methodologies described in these papers.

---

## Features

### Data Generation

- The `Matlab/` folder contains code that generates synthetic swarm-on-swarm engagement data, including position and velocity time series. This dataset serves as the input for training and evaluating neural network models.

### Neural Network Training and Evaluation

- The `class.py` script handles data preprocessing, model definition, training, and evaluation of neural network performance for swarm TSC tasks.

---

## How to Install and Use

### Installation

This project uses a Conda environment. To install the required dependencies:

```bash
conda env create -f swarm.yml
conda activate swarm
```

### Running the Code

This codebase is optimized for use on High-Performance Computing (HPC) environments managed by SLURM. SLURM job submission scripts (`.sh` files) are included to guide training and evaluation workflows.

---

## Disclaimer

All code in this repository is provided "as is", with no warranty (even implied) that it will work as advertised. No support is provided, and the authors or contributors are not responsible for any damages or losses resulting from use of this code.

---

## License

Copyright 2025 Donald Peltier

Licensed under the Apache License, Version 2.0 (the "License");  
you may not use this file except in compliance with the License.  
You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software  
distributed under the License is distributed on an "AS IS" BASIS,  
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
See the License for the specific language governing permissions and  
limitations under the License.

---

## <a name="citations"></a>Citations

1. D. W. Peltier, I. Kaminer, A. H. Clark, and M. Orescanin, “Swarm characteristics classification using neural networks,” *IEEE Transactions on Aerospace and Electronic Systems*, vol. 61, no. 1, pp. 389–400, Feb. 2025. Available: [https://doi.org/10.1109/TAES.2024.3447615](https://doi.org/10.1109/TAES.2024.3447615)

2. D. W. Peltier, I. Kaminer, A. H. Clark, and M. Orescanin, “Swarm characteristics classification using robust neural networks with optimized controllable inputs,” *arXiv preprint*, 2025. Available: [https://doi.org/10.48550/arXiv.2502.03619](https://doi.org/10.48550/arXiv.2502.03619)

3. D. W. Peltier, I. Kaminer, A. H. Clark, and M. Orescanin, “Swarm characteristics classification uncertainty using Bayesian neural networks,” April 2025. [Manuscript complete, submission pending].
