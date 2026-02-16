# Quantum-Enhanced Smart Computing Framework for Sustainable Credit Risk Decision Communication

<p align="center">
  <em>A Data-Intensive Science Project — Hybrid Quantum-Classical Machine Learning for Financial Risk Assessment</em>
</p>

---

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.39-blueviolet.svg)](https://pennylane.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository implements a **production-quality hybrid Quantum-Classical Credit Risk Prediction framework** based on the research paper:

> **"Quantum-Enhanced Smart Computing Framework for Sustainable Credit Risk Decision Communication"**
> — Jain, Singh, et al. (2026)

The framework demonstrates **quantum advantage** in financial risk classification by:

1. **Spectral Feature Engineering** — Extracting frequency-domain patterns from financial data using FFT and DCT transforms.
2. **Quantum Kernel Estimation** — Leveraging variational quantum circuits to compute kernel matrices in an exponentially large Hilbert space.
3. **Hybrid QSVM Classification** — Combining quantum kernel evaluation with classical SVM optimisation for binary credit risk prediction.

### Computational Complexity & Quantum Advantage

| Aspect | Classical SVM | Quantum SVM |
|--------|:------------:|:-----------:|
| Feature space dimension | O(d) | O(2ⁿ) |
| Kernel computation | Polynomial | Exponential Hilbert space |
| Feature correlations | Linear / RBF | Full entanglement-based |
| Target accuracy | ~90 % | **94.53 %** |

The quantum kernel maps *n*-dimensional classical data into a *2ⁿ*-dimensional Hilbert space via angle embedding and variational circuits, capturing feature correlations that are intractable for classical kernels.

## Architecture

```
┌───────────────┐     ┌──────────────────┐     ┌───────────────────┐
│ Raw Credit    │────▶│  Preprocessing   │────▶│  Spectral Feature │
│ Risk Data     │     │  (Eq. 1–3)       │     │  Engineering      │
└───────────────┘     └──────────────────┘     │  (FFT/DCT, Eq.5–6)│
                                                └────────┬──────────┘
                                                         │
                      ┌──────────────────┐               │
                      │  Classical       │◀──────────────┤
                      │  Baselines       │               │
                      │  (LR, RF, CNN)   │               │
                      └───────┬──────────┘               │
                              │                          ▼
                              │              ┌───────────────────────┐
                              │              │  Quantum Kernel       │
                              │              │  (PennyLane VQC)      │
                              │              │  Eq. 4, 7, 8          │
                              │              └────────┬──────────────┘
                              │                       │
                              │                       ▼
                              │              ┌───────────────────────┐
                              │              │  Quantum SVM          │
                              │              │  (sklearn SVC +       │
                              │              │   precomputed kernel) │
                              │              └────────┬──────────────┘
                              │                       │
                              ▼                       ▼
                      ┌──────────────────────────────────────┐
                      │        Evaluation & Comparison       │
                      │  Accuracy · Precision · Recall · F1  │
                      │  ROC-AUC · Confusion Matrix          │
                      └──────────────────────────────────────┘
```

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Data pipeline (Eq. 1–3)
│   ├── fft_dct_features.py       # Spectral feature engineering (Eq. 5–6)
│   ├── quantum_kernel.py         # PennyLane quantum kernel (Eq. 4, 7, 8)
│   ├── svm_classifier.py         # Hybrid Quantum SVM
│   ├── classical_baselines.py    # LR, Random Forest, CNN
│   ├── evaluation.py             # Metrics & visualisation
│   └── main.py                   # CLI orchestration entry point
├── notebooks/
│   └── quantum_credit_risk_analysis.ipynb
├── docs/
│   └── derivations.tex           # LaTeX mathematical derivations
├── results/                      # JSON evaluation outputs
├── visualizations/               # Generated plots
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Full pipeline with synthetic data (default — no Kaggle CSV needed)
python src/main.py --sample_size 200 --seed 42

# With a real dataset
python src/main.py --data_path data/credit_risk_dataset.csv --sample_size 300

# Minimal test run
python src/main.py --sample_size 50 --n_qubits 4 --cnn_epochs 10
```

### 3. CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data_path` | `None` | Path to credit risk CSV (uses synthetic fallback) |
| `--n_qubits` | `8` | Number of qubits for quantum kernel |
| `--n_layers` | `2` | Variational circuit layers |
| `--sample_size` | `200` | Training subset for quantum kernel |
| `--cnn_epochs` | `50` | CNN training epochs |
| `--seed` | `42` | Global random seed |
| `--output_dir` | `results/` | JSON output directory |
| `--viz_dir` | `visualizations/` | Plot output directory |

## Mathematical Foundations

The framework implements the following equations from the paper:

| Equation | Description | Module |
|----------|-------------|--------|
| Eq. 1 | Mean Imputation | `preprocessing.py` |
| Eq. 2 | Min-Max Normalisation → [0, 1] | `preprocessing.py` |
| Eq. 3 | One-Hot Encoding | `preprocessing.py` |
| Eq. 4 | Angle Embedding (R_Y rotations) | `quantum_kernel.py` |
| Eq. 5 | FFT Power Spectrum | `fft_dct_features.py` |
| Eq. 6 | DCT-II Energy Compaction | `fft_dct_features.py` |
| Eq. 7 | Variational Circuit (R_X/R_Z + CNOT) | `quantum_kernel.py` |
| Eq. 8 | Quantum Kernel K(xᵢ, xⱼ) = \|⟨φ(xᵢ)\|φ(xⱼ)⟩\|² | `quantum_kernel.py` |

## Citation

```bibtex
@article{jain2026quantum,
  title   = {Quantum-Enhanced Smart Computing Framework for Sustainable
             Credit Risk Decision Communication},
  author  = {Jain, A. and Singh, R. and others},
  journal = {Journal of Quantum Computing and Finance},
  year    = {2026}
}
```

## License

This project is released under the MIT License.
