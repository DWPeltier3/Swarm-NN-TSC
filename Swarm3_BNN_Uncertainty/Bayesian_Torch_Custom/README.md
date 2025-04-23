# Custom Bayesian-Torch (v0.5.0)

This repository contains a customized version of [Bayesian-Torch](https://github.com/IntelLabs/bayesian-torch) v0.5.0, with bug fixes and minor adjustments for compatibility with specific environments.

The original license, metadata, and authorship have been preserved.

---

## Contents

- `bayesian_torch/` – Modified source code with fixes.
- `bayesian_torch-0.5.0.dist-info/` – Original distribution metadata, including LICENSE.

---

## How to Use

1. Clone or download this repository.
2. Copy the `bayesian_torch/` folder into your Python environment's `site-packages/` directory.
3. (Optional) Copy the `bayesian_torch-0.5.0.dist-info/` folder to preserve version metadata.

---

## Notes

- This version is based on the official Bayesian-Torch v0.5.0 release, but includes custom bug fixes specific to the `swarm` Conda environment (Python 3.9, PyTorch compatibility).
- The `RECORD` file inside the `.dist-info` folder may not exactly match the modified source files. This is expected after local changes.
- Please refer to the LICENSE file inside the `bayesian_torch-0.5.0.dist-info/` folder for original licensing terms.

---

## Attribution

Bayesian-Torch is developed and maintained by Intel Labs.  
For the official version, please visit: [https://github.com/IntelLabs/bayesian-torch](https://github.com/IntelLabs/bayesian-torch).
