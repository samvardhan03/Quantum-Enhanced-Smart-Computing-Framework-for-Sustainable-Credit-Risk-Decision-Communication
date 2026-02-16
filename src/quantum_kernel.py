"""
Quantum Kernel Module
=====================
Implements the quantum feature-map and kernel computation for credit risk
classification, as described in:

    "Quantum-Enhanced Smart Computing Framework for Sustainable Credit Risk
     Decision Communication" (Jain, Singh, et al., 2026)

This module provides:
    - Angle Embedding        (Eq. 4)
    - Variational Circuit    (Eq. 7)  — R_X / R_Z rotations + CNOT entanglers
    - Quantum Kernel         (Eq. 8)  — K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²
    - Kernel Matrix builder  for use with scikit-learn SVC(kernel='precomputed')

Backend: PennyLane ``default.qubit`` simulator.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pennylane as qml


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_N_QUBITS: int = 12
DEFAULT_N_LAYERS: int = 2


# ---------------------------------------------------------------------------
# Quantum feature map components
# ---------------------------------------------------------------------------

def angle_embedding(features: np.ndarray, wires: list[int]) -> None:
    """Map classical feature vector onto qubits via R_Y rotations.

    Implements **Eq. 4** from the paper:

        |φ(x)⟩ = ⊗_{i=1}^{n} R_Y(x_i) |0⟩

    Each feature x_i ∈ [0, 1] is encoded as the rotation angle of an R_Y gate
    on the i-th qubit.  Features are scaled by π to use the full Bloch-sphere
    range: θ_i = π · x_i.

    Parameters
    ----------
    features : np.ndarray
        1-D array of normalised features with ``len(features) <= len(wires)``.
    wires : list[int]
        Qubit indices to embed onto.

    Notes
    -----
    If ``len(features) < len(wires)``, only the first ``len(features)`` qubits
    receive a rotation; the remaining qubits stay in |0⟩.
    """
    n_features = min(len(features), len(wires))
    for i in range(n_features):
        qml.RY(np.pi * features[i], wires=wires[i])


def variational_circuit(
    params: np.ndarray,
    wires: list[int],
    n_layers: int = DEFAULT_N_LAYERS,
) -> None:
    """Apply a hardware-efficient variational ansatz.

    Implements **Eq. 7** from the paper — a layered circuit comprising:

        1. Single-qubit R_X(θ) and R_Z(φ) rotations on every qubit.
        2. A ring of CNOT gates for nearest-neighbour entanglement.

    One *layer* consists of steps (1) and (2).  The ``params`` tensor has shape
    ``(n_layers, n_qubits, 2)`` where the last axis indexes R_X and R_Z
    angles respectively.

    Parameters
    ----------
    params : np.ndarray
        Variational parameters, shape ``(n_layers, len(wires), 2)``.
    wires : list[int]
        Qubit indices.
    n_layers : int, default 2
        Number of variational layers.
    """
    n_qubits = len(wires)
    for layer in range(n_layers):
        # Single-qubit rotations
        for q in range(n_qubits):
            qml.RX(params[layer, q, 0], wires=wires[q])
            qml.RZ(params[layer, q, 1], wires=wires[q])
        # CNOT entangling ring
        for q in range(n_qubits):
            qml.CNOT(wires=[wires[q], wires[(q + 1) % n_qubits]])


# ---------------------------------------------------------------------------
# Quantum kernel circuit
# ---------------------------------------------------------------------------

def _build_kernel_circuit(
    n_qubits: int = DEFAULT_N_QUBITS,
    n_layers: int = DEFAULT_N_LAYERS,
) -> Callable[..., float]:
    """Build a PennyLane QNode that evaluates the quantum kernel entry.

    Implements **Eq. 8** from the paper:

        K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²

    The circuit:
        1. Applies the feature map U(x_i) = VQC · AngleEmbedding(x_i).
        2. Applies U†(x_j).
        3. Measures the probability of the all-zero state |0…0⟩.

    The probability of measuring |0…0⟩ equals |⟨φ(x_i)|φ(x_j)⟩|² by
    construction.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of variational layers.

    Returns
    -------
    callable
        A QNode mapping ``(x1, x2, params) → float``.
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(dev, interface="numpy")
    def kernel_circuit(
        x1: np.ndarray,
        x2: np.ndarray,
        params: np.ndarray,
    ) -> float:
        """Evaluate K(x1, x2) = |⟨φ(x1)|φ(x2)⟩|²  (Eq. 8)."""
        # U(x1)
        angle_embedding(x1, wires)
        variational_circuit(params, wires, n_layers)
        # U†(x2)  — apply inverse in reverse order
        qml.adjoint(variational_circuit)(params, wires, n_layers)
        qml.adjoint(angle_embedding)(x2, wires)
        # Probability of |0…0⟩ — return full vector; extract [0] after execution
        return qml.probs(wires=wires)

    return kernel_circuit


# ---------------------------------------------------------------------------
# Public API: QuantumKernel class
# ---------------------------------------------------------------------------

class QuantumKernel:
    """Quantum kernel evaluator for use with scikit-learn SVC.

    Wraps the PennyLane quantum kernel circuit (Eq. 8) and provides methods to:

    * Evaluate a single kernel entry  ``K(x_i, x_j)``.
    * Compute the full Gram matrix for a dataset.
    * Compute the rectangular kernel matrix between train and test sets.

    Parameters
    ----------
    n_qubits : int, default 12
        Number of qubits in the circuit.
    n_layers : int, default 2
        Number of variational layers in the ansatz.
    params : np.ndarray, optional
        Pre-trained variational parameters.  If *None*, random parameters are
        initialised with seed ``random_state``.
    random_state : int, default 42
        RNG seed for parameter initialisation.

    Examples
    --------
    >>> import numpy as np
    >>> qk = QuantumKernel(n_qubits=4, n_layers=1)
    >>> x = np.random.rand(4)
    >>> qk.evaluate(x, x)  # should be ≈ 1.0
    """

    def __init__(
        self,
        n_qubits: int = DEFAULT_N_QUBITS,
        n_layers: int = DEFAULT_N_LAYERS,
        params: Optional[np.ndarray] = None,
        random_state: int = 42,
    ) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        if params is not None:
            self.params = params
        else:
            rng = np.random.default_rng(random_state)
            self.params = rng.uniform(
                0, 2 * np.pi, size=(n_layers, n_qubits, 2)
            )

        self._circuit = _build_kernel_circuit(n_qubits, n_layers)

    # ---- single entry ----

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute a single quantum kernel entry K(x1, x2).

        Implements **Eq. 8**:
            K(x_i, x_j) = |⟨φ(x_i)|φ(x_j)⟩|²

        Parameters
        ----------
        x1, x2 : np.ndarray
            Feature vectors of length ``<= n_qubits``, normalised to [0, 1].

        Returns
        -------
        float
            Kernel value in [0, 1].
        """
        x1 = np.asarray(x1, dtype=np.float64)
        x2 = np.asarray(x2, dtype=np.float64)
        probs = self._circuit(x1, x2, self.params)
        return float(probs[0])  # P(|0…0⟩) = K(x1, x2)

    # ---- Gram matrix (train) ----

    def compute_kernel_matrix(
        self,
        X: np.ndarray,
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute the symmetric Gram (kernel) matrix for dataset X.

        K_{ij} = K(x_i, x_j)   for all pairs  (Eq. 8).

        Parameters
        ----------
        X : np.ndarray
            Dataset of shape ``(n_samples, n_features)``.
        verbose : bool, default False
            Print progress every 10 % of pairs computed.

        Returns
        -------
        np.ndarray
            Symmetric kernel matrix of shape ``(n_samples, n_samples)``.
        """
        n = len(X)
        K = np.zeros((n, n), dtype=np.float64)
        total_pairs = n * (n + 1) // 2
        computed = 0

        for i in range(n):
            for j in range(i, n):
                val = self.evaluate(X[i], X[j])
                K[i, j] = val
                K[j, i] = val
                computed += 1
                if verbose and computed % max(1, total_pairs // 10) == 0:
                    print(
                        f"  Kernel matrix: {computed}/{total_pairs} pairs "
                        f"({100 * computed / total_pairs:.0f}%)"
                    )

        return K

    # ---- rectangular kernel (test vs train) ----

    def compute_kernel_matrix_test(
        self,
        X_test: np.ndarray,
        X_train: np.ndarray,
        verbose: bool = False,
    ) -> np.ndarray:
        """Compute the rectangular kernel matrix between test and train sets.

        K_{ij} = K(x_test_i, x_train_j)   (Eq. 8).

        Parameters
        ----------
        X_test : np.ndarray
            Test set, shape ``(n_test, n_features)``.
        X_train : np.ndarray
            Training set, shape ``(n_train, n_features)``.
        verbose : bool, default False
            Print progress.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape ``(n_test, n_train)``.
        """
        n_test, n_train = len(X_test), len(X_train)
        K = np.zeros((n_test, n_train), dtype=np.float64)
        total = n_test * n_train
        computed = 0

        for i in range(n_test):
            for j in range(n_train):
                K[i, j] = self.evaluate(X_test[i], X_train[j])
                computed += 1
                if verbose and computed % max(1, total // 10) == 0:
                    print(
                        f"  Test kernel: {computed}/{total} pairs "
                        f"({100 * computed / total:.0f}%)"
                    )

        return K

    # ---- scikit-learn compatible callable ----

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute the kernel matrix between two sets (sklearn-compatible).

        This allows the kernel to be passed directly to
        ``sklearn.svm.SVC(kernel=quantum_kernel_instance)``.

        Parameters
        ----------
        X1, X2 : np.ndarray
            Feature matrices.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape ``(len(X1), len(X2))``.
        """
        if np.array_equal(X1, X2):
            return self.compute_kernel_matrix(X1)
        return self.compute_kernel_matrix_test(X1, X2)


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Quantum Kernel Module — Self-Test")
    print("=" * 60)

    N_QUBITS_TEST = 4
    N_LAYERS_TEST = 1

    qk = QuantumKernel(
        n_qubits=N_QUBITS_TEST,
        n_layers=N_LAYERS_TEST,
        random_state=42,
    )

    # Test 1: K(x, x) ≈ 1
    x = np.random.default_rng(0).uniform(0, 1, size=N_QUBITS_TEST)
    kxx = qk.evaluate(x, x)
    print(f"\nK(x, x) = {kxx:.6f}  (expected ≈ 1.0)")
    assert abs(kxx - 1.0) < 1e-4, f"Self-kernel failed: {kxx}"

    # Test 2: symmetry K(x, y) == K(y, x)
    y = np.random.default_rng(1).uniform(0, 1, size=N_QUBITS_TEST)
    kxy = qk.evaluate(x, y)
    kyx = qk.evaluate(y, x)
    print(f"K(x, y) = {kxy:.6f}")
    print(f"K(y, x) = {kyx:.6f}  (should equal K(x, y))")
    assert abs(kxy - kyx) < 1e-6, "Symmetry check failed"

    # Test 3: small kernel matrix
    X_small = np.random.default_rng(2).uniform(0, 1, size=(5, N_QUBITS_TEST))
    K_mat = qk.compute_kernel_matrix(X_small, verbose=True)
    print(f"\nKernel matrix shape: {K_mat.shape}")
    print(f"Diagonal (should be ≈ 1): {np.diag(K_mat)}")
    assert np.allclose(np.diag(K_mat), 1.0, atol=1e-4), "Diagonal check failed"

    print("\n✓ All quantum kernel tests passed.")
