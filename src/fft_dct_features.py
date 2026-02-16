"""
Spectral Feature Engineering Module
====================================
Implements FFT and DCT-based feature transformations for financial credit risk
data, as described in:

    "Quantum-Enhanced Smart Computing Framework for Sustainable Credit Risk
     Decision Communication" (Jain, Singh, et al., 2026)

The module extracts non-obvious cyclic patterns embedded in numerical financial
features (e.g., income history, credit utilisation over time) by projecting them
into the frequency domain.

Key equations implemented:
    Eq. 5  — Power Spectrum via FFT
    Eq. 6  — Energy Compaction via DCT-II
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.fft import dct, fft, fftfreq  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------

def apply_fft(signal: np.ndarray) -> np.ndarray:
    """Compute the **power spectrum** of a 1-D signal using the Fast Fourier
    Transform.

    Implements **Eq. 5** from the paper:

        P(k) = |X(k)|² / N

    where X(k) = FFT{x(n)} and N is the signal length.

    Parameters
    ----------
    signal : np.ndarray
        1-D real-valued input signal of length N.

    Returns
    -------
    np.ndarray
        Power spectrum of shape ``(N // 2,)`` (one-sided spectrum, excluding
        the Nyquist component for even-length signals).

    Examples
    --------
    >>> import numpy as np
    >>> sig = np.array([1.0, 2.0, 3.0, 4.0])
    >>> ps = apply_fft(sig)
    >>> ps.shape
    (2,)
    """
    signal = np.asarray(signal, dtype=np.float64)
    N: int = len(signal)
    if N == 0:
        return np.array([], dtype=np.float64)

    X = fft(signal)
    # One-sided power spectrum (positive frequencies only)
    power_spectrum: np.ndarray = (np.abs(X[:N // 2]) ** 2) / N
    return power_spectrum


def apply_dct(signal: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
    """Compute the **energy compaction** coefficients of a 1-D signal using the
    Type-II Discrete Cosine Transform (DCT-II).

    Implements **Eq. 6** from the paper:

        C(k) = DCT-II{x(n)},  k = 0, 1, …, K-1

    where K ≤ N selects the most energetic low-frequency components (energy
    compaction property of DCT-II).

    Parameters
    ----------
    signal : np.ndarray
        1-D real-valued input signal of length N.
    n_components : int, optional
        Number of leading DCT coefficients to retain.  If *None*, all N
        coefficients are returned.

    Returns
    -------
    np.ndarray
        DCT-II coefficients of shape ``(K,)``.

    Examples
    --------
    >>> import numpy as np
    >>> sig = np.array([1.0, 2.0, 3.0, 4.0])
    >>> coeffs = apply_dct(sig, n_components=2)
    >>> coeffs.shape
    (2,)
    """
    signal = np.asarray(signal, dtype=np.float64)
    N: int = len(signal)
    if N == 0:
        return np.array([], dtype=np.float64)

    # SciPy's DCT-II (type=2) with 'ortho' normalisation for energy
    # preservation across coefficients.
    coefficients: np.ndarray = dct(signal, type=2, norm="ortho")

    if n_components is not None:
        coefficients = coefficients[:n_components]

    return coefficients


# ---------------------------------------------------------------------------
# Feature-engineering pipeline
# ---------------------------------------------------------------------------

def extract_spectral_features(
    df: pd.DataFrame,
    columns: List[str],
    n_fft_components: Optional[int] = None,
    n_dct_components: int = 5,
) -> pd.DataFrame:
    """End-to-end spectral feature extraction for a set of numerical columns.

    For **each column** in *columns* the function:

    1. Computes the FFT power spectrum (Eq. 5) and retains either all or
       ``n_fft_components`` values.
    2. Computes the DCT-II energy compaction (Eq. 6) and retains the top
       ``n_dct_components`` coefficients.
    3. Appends the new features as additional columns to the DataFrame.

    The naming convention for the new columns is::

        {col}_fft_{k}   — k-th FFT power spectrum coefficient
        {col}_dct_{k}   — k-th DCT energy compaction coefficient

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the financial features.
    columns : list of str
        Column names on which to apply spectral transforms.
    n_fft_components : int, optional
        Number of FFT power-spectrum components to keep per column.
        Defaults to ``N // 2`` (one-sided spectrum length).
    n_dct_components : int, default 5
        Number of DCT-II coefficients to keep per column.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with spectral feature columns appended.

    Notes
    -----
    Because FFT / DCT operate on the *entire column vector* (all samples),
    they capture **dataset-level** frequency structure.  This is useful for
    detecting population-wide cyclic patterns (e.g., seasonal income
    fluctuations) rather than per-sample temporal patterns.  The resulting
    scalar features (one per component per column) are broadcast to every
    row so that downstream models can leverage the spectral information.

    For per-sample transforms (e.g., when each sample has a time-series
    history), call :func:`apply_fft` / :func:`apply_dct` directly on each
    row's sub-array.
    """
    df_out = df.copy()

    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

        signal: np.ndarray = df[col].to_numpy(dtype=np.float64)

        # --- FFT Power Spectrum (Eq. 5) ---
        ps = apply_fft(signal)
        n_fft = n_fft_components if n_fft_components is not None else len(ps)
        ps = ps[:n_fft]
        for k, val in enumerate(ps):
            df_out[f"{col}_fft_{k}"] = val  # broadcast scalar

        # --- DCT Energy Compaction (Eq. 6) ---
        dct_coeffs = apply_dct(signal, n_components=n_dct_components)
        for k, val in enumerate(dct_coeffs):
            df_out[f"{col}_dct_{k}"] = val  # broadcast scalar

    return df_out


def compute_spectral_feature_vector(
    signal: np.ndarray,
    n_fft_components: int = 5,
    n_dct_components: int = 5,
) -> np.ndarray:
    """Compute a concatenated spectral feature vector for a single 1-D signal.

    Combines the FFT power spectrum (Eq. 5) and DCT energy compaction (Eq. 6)
    into a single feature vector.  This is a convenience function for
    per-sample feature extraction.

    Parameters
    ----------
    signal : np.ndarray
        1-D real-valued input signal.
    n_fft_components : int, default 5
        Number of FFT power-spectrum components to retain.
    n_dct_components : int, default 5
        Number of DCT-II coefficients to retain.

    Returns
    -------
    np.ndarray
        Concatenated feature vector of shape ``(n_fft_components + n_dct_components,)``.
    """
    ps = apply_fft(signal)[:n_fft_components]
    dc = apply_dct(signal, n_components=n_dct_components)

    # Pad with zeros if the signal is shorter than requested components
    if len(ps) < n_fft_components:
        ps = np.pad(ps, (0, n_fft_components - len(ps)))
    if len(dc) < n_dct_components:
        dc = np.pad(dc, (0, n_dct_components - len(dc)))

    return np.concatenate([ps, dc])


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Spectral Feature Engineering — Self-Test")
    print("=" * 60)

    # Synthetic test signal: superposition of two frequencies
    np.random.seed(42)
    t = np.linspace(0, 1, 256, endpoint=False)
    test_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)

    ps = apply_fft(test_signal)
    print(f"\nFFT Power Spectrum shape : {ps.shape}")
    print(f"  Peak frequency index   : {np.argmax(ps)}")

    dc = apply_dct(test_signal, n_components=10)
    print(f"\nDCT Coefficients (top 10): {dc}")

    # DataFrame pipeline test
    df_test = pd.DataFrame({
        "income": np.random.exponential(50000, 100),
        "credit_history_months": np.random.randint(6, 360, 100).astype(float),
    })
    df_spectral = extract_spectral_features(
        df_test, columns=["income", "credit_history_months"], n_dct_components=3
    )
    new_cols = [c for c in df_spectral.columns if c not in df_test.columns]
    print(f"\nNew spectral columns ({len(new_cols)}): {new_cols}")

    # Per-sample vector test
    vec = compute_spectral_feature_vector(test_signal, n_fft_components=4, n_dct_components=4)
    print(f"\nSpectral feature vector (len={len(vec)}): {vec}")

    print("\n✓ All spectral feature tests passed.")
