#!/usr/bin/env python3
"""
Main Orchestration Module
=========================
End-to-end pipeline for Quantum-Enhanced Credit Risk Prediction.

Usage::

    python src/main.py --sample_size 200 --seed 42

This script orchestrates:
    1. Data preprocessing (Eq. 1–3)
    2. Spectral feature engineering (Eq. 5–6)
    3. Quantum kernel computation & QSVM training (Eq. 4, 7, 8)
    4. Classical baseline training (LR, RF, CNN)
    5. Evaluation, comparison, and visualisation

Reference:
    "Quantum-Enhanced Smart Computing Framework for Sustainable Credit Risk
     Decision Communication" (Jain, Singh, et al., 2026)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on the path when running as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classical_baselines import evaluate_all_baselines
from src.evaluation import (
    compute_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curve,
    print_classification_report,
    save_results,
)
from src.fft_dct_features import extract_spectral_features
from src.preprocessing import preprocess_pipeline
from src.svm_classifier import QuantumSVM


# ═══════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantum-Enhanced Credit Risk Prediction Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to credit risk CSV.  Falls back to synthetic data if omitted.",
    )
    parser.add_argument(
        "--n_qubits",
        type=int,
        default=8,
        help="Number of qubits for the quantum kernel (≤ number of features).",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Number of variational layers in the quantum circuit.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=200,
        help="Subset size for quantum kernel computation (for tractability).",
    )
    parser.add_argument(
        "--cnn_epochs",
        type=int,
        default=50,
        help="Training epochs for the CNN baseline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
        help="Directory for JSON results.",
    )
    parser.add_argument(
        "--viz_dir",
        type=str,
        default=str(PROJECT_ROOT / "visualizations"),
        help="Directory for plots.",
    )
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    viz_dir = Path(args.viz_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Quantum-Enhanced Credit Risk Prediction Pipeline")
    print("=" * 70)
    print(f"  Qubits       : {args.n_qubits}")
    print(f"  VQC layers   : {args.n_layers}")
    print(f"  Sample size  : {args.sample_size}")
    print(f"  Seed         : {args.seed}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Preprocessing  (Eq. 1–3)
    # ------------------------------------------------------------------
    print("\n▶ Step 1: Data Preprocessing")
    X_train_full, X_test_full, y_train_full, y_test_full, feature_names = preprocess_pipeline(
        path=args.data_path,
        random_state=args.seed,
    )

    # ------------------------------------------------------------------
    # 2. Spectral Feature Engineering  (Eq. 5–6)
    # ------------------------------------------------------------------
    print("\n▶ Step 2: Spectral Feature Engineering (FFT / DCT)")
    import pandas as pd

    # Select numeric feature names for spectral extraction
    # Use up to 5 features for spectral analysis to keep dimensionality manageable
    n_spectral_cols = min(5, len(feature_names))
    spectral_cols = feature_names[:n_spectral_cols]

    df_train = pd.DataFrame(X_train_full, columns=feature_names)
    df_test = pd.DataFrame(X_test_full, columns=feature_names)

    N_FFT_COMPONENTS = 5   # fixed count → train & test get identical column counts
    N_DCT_COMPONENTS = 3

    df_train_spectral = extract_spectral_features(
        df_train, columns=spectral_cols,
        n_fft_components=N_FFT_COMPONENTS, n_dct_components=N_DCT_COMPONENTS,
    )
    df_test_spectral = extract_spectral_features(
        df_test, columns=spectral_cols,
        n_fft_components=N_FFT_COMPONENTS, n_dct_components=N_DCT_COMPONENTS,
    )

    X_train_full = df_train_spectral.values
    X_test_full = df_test_spectral.values
    feature_names_extended = list(df_train_spectral.columns)

    print(f"  Features after spectral augmentation: {X_train_full.shape[1]}")

    # ------------------------------------------------------------------
    # 3. Subset sampling for quantum tractability
    # ------------------------------------------------------------------
    sample_train = min(args.sample_size, len(X_train_full))
    sample_test = min(args.sample_size // 4, len(X_test_full))

    rng = np.random.default_rng(args.seed)
    train_idx = rng.choice(len(X_train_full), sample_train, replace=False)
    test_idx = rng.choice(len(X_test_full), sample_test, replace=False)

    X_train_q = X_train_full[train_idx]
    y_train_q = y_train_full[train_idx]
    X_test_q = X_test_full[test_idx]
    y_test_q = y_test_full[test_idx]

    # Trim features to n_qubits for the quantum kernel
    n_q = min(args.n_qubits, X_train_q.shape[1])
    X_train_q = X_train_q[:, :n_q]
    X_test_q = X_test_q[:, :n_q]

    print(f"\n  Quantum subset: train={sample_train}, test={sample_test}, "
          f"features={n_q}")

    # ------------------------------------------------------------------
    # 4. Quantum SVM  (Eq. 4, 7, 8)
    # ------------------------------------------------------------------
    print("\n▶ Step 3: Quantum SVM Training")
    t0 = time.time()
    qsvm = QuantumSVM(
        n_qubits=n_q,
        n_layers=args.n_layers,
        random_state=args.seed,
    )
    qsvm.fit(X_train_q, y_train_q, verbose=True)
    qsvm_preds = qsvm.predict(X_test_q, verbose=True)
    qsvm_proba = qsvm.predict_proba(X_test_q, verbose=False)
    qsvm_time = time.time() - t0
    print(f"  QSVM completed in {qsvm_time:.1f}s")

    # ------------------------------------------------------------------
    # 5. Classical Baselines
    # ------------------------------------------------------------------
    print("\n▶ Step 4: Classical Baselines")
    baseline_results = evaluate_all_baselines(
        X_train_full, X_test_full, y_train_full, y_test_full,
        cnn_epochs=args.cnn_epochs,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------------
    print("\n▶ Step 5: Evaluation & Comparison")
    all_metrics: dict = {}

    # QSVM metrics
    qsvm_metrics = compute_metrics(y_test_q, qsvm_preds, qsvm_proba)
    all_metrics["Quantum SVM"] = qsvm_metrics
    print_classification_report(y_test_q, qsvm_preds, "Quantum SVM")
    plot_confusion_matrix(
        y_test_q, qsvm_preds,
        save_path=viz_dir / "qsvm_confusion_matrix.png",
        model_name="Quantum SVM",
    )
    if qsvm_proba is not None:
        plot_roc_curve(
            y_test_q, qsvm_proba,
            save_path=viz_dir / "qsvm_roc_curve.png",
            model_name="Quantum SVM",
        )

    # Baseline metrics
    for name, res in baseline_results.items():
        bm = compute_metrics(y_test_full, res["predictions"], res["probabilities"])
        all_metrics[name] = bm
        print_classification_report(y_test_full, res["predictions"], name)
        safe_name = name.lower().replace(" ", "_")
        plot_confusion_matrix(
            y_test_full, res["predictions"],
            save_path=viz_dir / f"{safe_name}_confusion_matrix.png",
            model_name=name,
        )
        if res["probabilities"] is not None:
            plot_roc_curve(
                y_test_full, res["probabilities"],
                save_path=viz_dir / f"{safe_name}_roc_curve.png",
                model_name=name,
            )

    # Comparison chart
    plot_model_comparison(all_metrics, save_path=viz_dir / "model_comparison.png")

    # Save JSON results
    save_results(all_metrics, output_dir / "evaluation_results.json")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" RESULTS SUMMARY")
    print("=" * 70)
    for name, m in all_metrics.items():
        print(f"  {name:25s}  Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
              f"AUC={m.get('auc_roc', float('nan')):.4f}")
    print("=" * 70)
    print(f"\n  Results  → {output_dir}")
    print(f"  Plots    → {viz_dir}")
    print("  Done ✓\n")


if __name__ == "__main__":
    main()
