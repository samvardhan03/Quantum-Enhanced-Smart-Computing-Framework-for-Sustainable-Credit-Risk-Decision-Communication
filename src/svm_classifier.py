"""
Hybrid Quantum-Classical SVM Classifier
========================================
Integrates the PennyLane quantum kernel (Eq. 8) with a scikit-learn SVC to
build a Quantum Support Vector Machine (QSVM), as described in:

    "Quantum-Enhanced Smart Computing Framework for Sustainable Credit Risk
     Decision Communication" (Jain, Singh, et al., 2026)

The classifier accepts a pre-computed quantum kernel matrix for both training
and inference, enabling seamless hybrid quantum-classical workflows.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.svm import SVC

from .quantum_kernel import QuantumKernel


class QuantumSVM:
    """Quantum Support Vector Machine using a pre-computed quantum kernel.

    Wraps ``sklearn.svm.SVC(kernel='precomputed')`` and uses the
    :class:`QuantumKernel` evaluator to build Gram matrices.

    Parameters
    ----------
    kernel_fn : QuantumKernel or callable, optional
        A callable ``(X1, X2) → np.ndarray`` that computes the kernel matrix.
        If *None*, a default :class:`QuantumKernel` is instantiated.
    C : float, default 1.0
        SVM regularisation parameter.
    n_qubits : int, default 12
        Qubits for the default kernel (ignored if ``kernel_fn`` is provided).
    n_layers : int, default 2
        Variational layers (ignored if ``kernel_fn`` is provided).
    random_state : int, default 42
        Seed for the default kernel parameter initialisation.

    Examples
    --------
    >>> import numpy as np
    >>> from src.svm_classifier import QuantumSVM
    >>> qsvm = QuantumSVM(n_qubits=4, n_layers=1)
    >>> X_train = np.random.rand(20, 4)
    >>> y_train = np.random.randint(0, 2, 20)
    >>> qsvm.fit(X_train, y_train)
    >>> preds = qsvm.predict(X_train[:5])
    """

    def __init__(
        self,
        kernel_fn: Optional[Callable[..., np.ndarray] | QuantumKernel] = None,
        C: float = 1.0,
        n_qubits: int = 12,
        n_layers: int = 2,
        random_state: int = 42,
    ) -> None:
        if kernel_fn is not None:
            self.kernel = kernel_fn
        else:
            self.kernel = QuantumKernel(
                n_qubits=n_qubits,
                n_layers=n_layers,
                random_state=random_state,
            )

        self._svc = SVC(kernel="precomputed", C=C, probability=True, random_state=random_state)
        self._X_train: Optional[np.ndarray] = None

    # ---- Training ----

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True,
    ) -> "QuantumSVM":
        """Train the QSVM.

        Builds the quantum kernel Gram matrix K(X_train, X_train) and fits the
        SVC on it.

        Parameters
        ----------
        X_train : np.ndarray
            Training features, shape ``(n_train, n_features)``.
        y_train : np.ndarray
            Training labels, shape ``(n_train,)``.
        verbose : bool, default True
            Print progress during kernel computation.

        Returns
        -------
        QuantumSVM
            Self (for chaining).
        """
        self._X_train = X_train.copy()

        if verbose:
            print("[QSVM] Computing training kernel matrix …")
        if isinstance(self.kernel, QuantumKernel):
            K_train = self.kernel.compute_kernel_matrix(X_train, verbose=verbose)
        else:
            K_train = self.kernel(X_train, X_train)

        if verbose:
            print("[QSVM] Fitting SVC on pre-computed kernel …")
        self._svc.fit(K_train, y_train)

        if verbose:
            print("[QSVM] Training complete.")
        return self

    # ---- Prediction ----

    def predict(
        self,
        X_test: np.ndarray,
        verbose: bool = False,
    ) -> np.ndarray:
        """Predict class labels for test samples.

        Builds the rectangular kernel matrix K(X_test, X_train) and delegates
        to the fitted SVC.

        Parameters
        ----------
        X_test : np.ndarray
            Test features, shape ``(n_test, n_features)``.
        verbose : bool, default False
            Print progress.

        Returns
        -------
        np.ndarray
            Predicted labels, shape ``(n_test,)``.
        """
        if self._X_train is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        if isinstance(self.kernel, QuantumKernel):
            K_test = self.kernel.compute_kernel_matrix_test(
                X_test, self._X_train, verbose=verbose,
            )
        else:
            K_test = self.kernel(X_test, self._X_train)

        return self._svc.predict(K_test)

    # ---- Probability estimates ----

    def predict_proba(
        self,
        X_test: np.ndarray,
        verbose: bool = False,
    ) -> np.ndarray:
        """Predict class probabilities for test samples.

        Parameters
        ----------
        X_test : np.ndarray
            Test features.
        verbose : bool, default False
            Print progress.

        Returns
        -------
        np.ndarray
            Class probabilities, shape ``(n_test, n_classes)``.
        """
        if self._X_train is None:
            raise RuntimeError("Model has not been fitted yet. Call .fit() first.")

        if isinstance(self.kernel, QuantumKernel):
            K_test = self.kernel.compute_kernel_matrix_test(
                X_test, self._X_train, verbose=verbose,
            )
        else:
            K_test = self.kernel(X_test, self._X_train)

        return self._svc.predict_proba(K_test)

    # ---- Convenience ----

    def score(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = False,
    ) -> float:
        """Return accuracy on a test set.

        Parameters
        ----------
        X_test : np.ndarray
            Test features.
        y_test : np.ndarray
            True labels.
        verbose : bool, default False
            Print progress.

        Returns
        -------
        float
            Classification accuracy.
        """
        preds = self.predict(X_test, verbose=verbose)
        return float(np.mean(preds == y_test))


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Quantum SVM Classifier — Self-Test")
    print("=" * 60)

    from .quantum_kernel import QuantumKernel as QK

    N_QUBITS = 4
    N_TRAIN = 20
    N_TEST = 5

    rng = np.random.default_rng(42)
    X_train = rng.uniform(0, 1, (N_TRAIN, N_QUBITS))
    y_train = rng.integers(0, 2, N_TRAIN)
    X_test = rng.uniform(0, 1, (N_TEST, N_QUBITS))
    y_test = rng.integers(0, 2, N_TEST)

    qsvm = QuantumSVM(n_qubits=N_QUBITS, n_layers=1)
    qsvm.fit(X_train, y_train, verbose=True)

    acc = qsvm.score(X_test, y_test, verbose=True)
    print(f"\nTest accuracy: {acc:.2%}")

    proba = qsvm.predict_proba(X_test)
    print(f"Probabilities shape: {proba.shape}")

    print("\n✓ QSVM self-test passed.")
