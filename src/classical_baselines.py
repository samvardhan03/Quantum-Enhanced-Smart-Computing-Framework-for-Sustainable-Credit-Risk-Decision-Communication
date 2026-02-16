"""
Classical Baseline Models
=========================
Implements Logistic Regression, Random Forest, and a 1-D CNN for rigorous
comparison against the Quantum SVM, as described in:

    "Quantum-Enhanced Smart Computing Framework for Sustainable Credit Risk
     Decision Communication" (Jain, Singh, et al., 2026)

All models expose a unified ``train_*`` / ``predict_*`` interface and return
numpy arrays for downstream evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset


# ═══════════════════════════════════════════════════════════════════════════
# Logistic Regression
# ═══════════════════════════════════════════════════════════════════════════

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Train a Logistic Regression classifier.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data.
    max_iter : int, default 1000
        Maximum solver iterations.
    random_state : int, default 42
        RNG seed.

    Returns
    -------
    LogisticRegression
        Fitted model.
    """
    model = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Random Forest
# ═══════════════════════════════════════════════════════════════════════════

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 12,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Train a Random Forest classifier.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data.
    n_estimators : int, default 200
        Number of trees.
    max_depth : int, default 12
        Maximum tree depth.
    random_state : int, default 42
        RNG seed.

    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 1-D Convolutional Neural Network (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════

class CreditRiskCNN(nn.Module):
    """Lightweight 1-D CNN for tabular credit risk classification.

    Architecture:
        Conv1d(1, 32, k=3) → ReLU → Conv1d(32, 64, k=3) → ReLU →
        AdaptiveAvgPool1d(1) → Flatten → FC(64, 32) → ReLU → Dropout → FC(32, 2)
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 1, n_features)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, 2)``.
        """
        return self.classifier(self.conv_block(x))


def build_cnn_model(input_dim: int) -> CreditRiskCNN:
    """Instantiate the 1-D CNN model.

    Parameters
    ----------
    input_dim : int
        Number of input features.

    Returns
    -------
    CreditRiskCNN
        Un-trained model.
    """
    return CreditRiskCNN(input_dim)


def train_cnn(
    model: CreditRiskCNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    verbose: bool = True,
) -> CreditRiskCNN:
    """Train the 1-D CNN using Adam and CrossEntropyLoss.

    Parameters
    ----------
    model : CreditRiskCNN
        Model to train.
    X_train : np.ndarray
        Training features, shape ``(n_samples, n_features)``.
    y_train : np.ndarray
        Training labels.
    epochs : int, default 50
        Training epochs.
    batch_size : int, default 32
        Mini-batch size.
    lr : float, default 1e-3
        Learning rate.
    verbose : bool, default True
        Print loss every 10 epochs.

    Returns
    -------
    CreditRiskCNN
        Trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Reshape to (N, 1, D) for Conv1d
    X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            avg_loss = epoch_loss / len(X_train)
            print(f"  [CNN] Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}")

    return model


def predict_cnn(
    model: CreditRiskCNN,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run CNN inference.

    Parameters
    ----------
    model : CreditRiskCNN
        Trained model.
    X : np.ndarray
        Features, shape ``(n_samples, n_features)``.

    Returns
    -------
    predictions : np.ndarray
        Predicted labels.
    probabilities : np.ndarray
        Class probabilities, shape ``(n_samples, 2)``.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs


# ═══════════════════════════════════════════════════════════════════════════
# Unified evaluation helper
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_all_baselines(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cnn_epochs: int = 50,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Train and evaluate all classical baselines.

    Returns a dictionary keyed by model name with sub-keys:
    ``"predictions"``, ``"probabilities"``, ``"model"``.

    Parameters
    ----------
    X_train, X_test, y_train, y_test : np.ndarray
        Pre-processed data.
    cnn_epochs : int, default 50
        Training epochs for the CNN.
    verbose : bool, default True
        Print training progress.

    Returns
    -------
    dict
        Results per model.
    """
    results: Dict[str, Dict[str, Any]] = {}

    # --- Logistic Regression ---
    if verbose:
        print("\n[Baselines] Training Logistic Regression …")
    lr_model = train_logistic_regression(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)
    results["Logistic Regression"] = {
        "predictions": lr_preds,
        "probabilities": lr_proba,
        "model": lr_model,
    }

    # --- Random Forest ---
    if verbose:
        print("[Baselines] Training Random Forest …")
    rf_model = train_random_forest(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)
    results["Random Forest"] = {
        "predictions": rf_preds,
        "probabilities": rf_proba,
        "model": rf_model,
    }

    # --- CNN ---
    if verbose:
        print("[Baselines] Training CNN …")
    cnn_model = build_cnn_model(X_train.shape[1])
    cnn_model = train_cnn(cnn_model, X_train, y_train, epochs=cnn_epochs, verbose=verbose)
    cnn_preds, cnn_proba = predict_cnn(cnn_model, X_test)
    results["CNN"] = {
        "predictions": cnn_preds,
        "probabilities": cnn_proba,
        "model": cnn_model,
    }

    return results


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Classical Baselines — Self-Test")
    print("=" * 60)

    rng = np.random.default_rng(42)
    N, D = 200, 10
    X_train = rng.uniform(0, 1, (N, D))
    y_train = rng.integers(0, 2, N)
    X_test = rng.uniform(0, 1, (50, D))
    y_test = rng.integers(0, 2, 50)

    results = evaluate_all_baselines(
        X_train, X_test, y_train, y_test, cnn_epochs=10, verbose=True,
    )

    for name, res in results.items():
        acc = np.mean(res["predictions"] == y_test)
        print(f"  {name:25s}  accuracy = {acc:.2%}")

    print("\n✓ Classical baselines self-test passed.")
