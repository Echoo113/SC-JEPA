<div align="center">

# SC-JEPA: Stabilizing Latent Predictive Learning for Time-Series Anomaly Prediction

**Yanan He¹ · Yunshi Wen² · Xin Wang³ · Tengfei Ma³,\***

¹Yale University &nbsp;&nbsp; ²Rensselaer Polytechnic Institute &nbsp;&nbsp; ³Stony Brook University

\*Corresponding author: [tengfei.ma@stonybrook.edu](mailto:tengfei.ma@stonybrook.edu)

**SIAM International Conference on Data Mining (SDM) 2026**

[Citation](#citation) · [Data](#data) · [Quick Start](#quick-start)

</div>

---

## Abstract

Time-series anomaly prediction aims to forecast future system failures before they fully emerge, making latent predictive models such as JEPA a promising framework for capturing precursor dynamics. However, directly applying continuous self-distillation to time-series data is often unstable and can lead to representation collapse, while also struggling to model precursors evolving at different temporal scales. To address this, we propose **SC-JEPA**, a new JEPA-based framework to model time-series anomaly prediction in a discretized predictive state space. It introduces a soft codebook bottleneck to stabilize latent predictive learning and encourage regime-level structure in the learned representations. Building on this stabilized latent space, we further design a multi-resolution predictive objective to capture precursor patterns at different temporal scales. Experiments on five real-world benchmarks show that SC-JEPA achieves strong and consistent early-warning performance.

---

## Method

- **Encoder** (`models/encoder.py`) — CNN/residual feature extractor mapping raw patches `(B, C, L)` into a latent space `(B, D)`.
- **Quantizer** (`models/quantizer.py`) — soft vector quantization over a learned codebook, producing per-patch code distributions `p`.
- **Predictor** (`models/predictor.py`) — Transformer that predicts future code-distribution logits and latent codes from past code distributions, at both fine (per-patch) and coarse (aggregated) resolutions.
- **Decoder** (`models/decoder.py`) — reconstructs patch-level signals from quantized latents for auxiliary supervision.
- **Objective** (`engine/losses.py`) — fine/coarse KL prediction losses, latent MSE alignment, VQ quantization + commitment losses, and sample-/batch-wise entropy regularization.
- **Downstream** (`downstream/`) — classifier and evaluation utilities that turn predicted code distributions into anomaly-prediction scores.

---

## Quick Start

**Requirements**
- Python ≥ 3.9
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

**Representative run**

```bash
# Train SC-JEPA
python train.py

# Evaluate downstream anomaly prediction
python eval.py --use_probs
```

---

## Data

**This repository does not include any dataset.** The `data/` folder only contains preprocessing code — you must download the raw data yourself from the official sources below before training or evaluating, and convert it into patch-level inputs with `data/preprocess.py`.

| Dataset | Source |
|---|---|
| MSL & SMAP (NASA) | [NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) |
| SWaT | [iTrust dataset portal](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/) |
| PSM | [eBay/Pyraformer](https://github.com/eBay/Pyraformer) |
| SMD | [NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly) |

**Expected input format**

- `x_patches`: shape `(N, P, L, V)`
- `y_label`: shape `(N,)` or `(N, P)`

All experiments assume `P = 5` patches per window and patch length `L = 20` (see constants in `train.py`).

**Preparing a dataset**

Run the raw data for each benchmark through `data/preprocess.py` to produce normalized, patch-level `.npz` files (`train_norm.npz` / `test_norm.npz`), then point `DATA_DIR` in `train.py` / `eval.py` to that folder.

---

## Repository Structure

```
train.py                  main training entrypoint (SC-JEPA)
eval.py                   evaluation / inference
models/
  encoder.py              CNN patch encoder
  quantizer.py            soft vector-quantization codebook
  predictor.py            Transformer latent predictor
  decoder.py              patch reconstruction decoder
engine/
  losses.py               KL, alignment, VQ, and entropy losses
  validate.py             validation loop
downstream/
  classifier.py           anomaly-prediction classifier
  train_classifier.py     downstream training script
  evaluation.py           downstream metrics
  features.py             feature extraction utilities
data/
  datasets.py             dataset wrappers
  preprocess.py           raw-to-patch preprocessing
  utils.py                data utilities
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{he2026scjepa,
  title     = {SC-JEPA: Stabilizing Latent Predictive Learning for Time-Series Anomaly Prediction},
  author    = {He, Yanan and Wen, Yunshi and Wang, Xin and Ma, Tengfei},
  booktitle = {Proceedings of the SIAM International Conference on Data Mining (SDM)},
  year      = {2026}
}
```

## License

© 2026 SIAM. Unauthorized reproduction of this article is prohibited.
