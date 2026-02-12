"""
Fractional Fourier Transform (FrFT) implementation.
Based on: Ozaktas, Arikan, Kutay, Bozdagi (1996)
"Digital Computation of the Fractional Fourier Transform"
IEEE Trans. Signal Processing, 44(9):2141-2150.
"""

import numpy as np


def _frft_core(f: np.ndarray, a: float) -> np.ndarray:
    """
    Compute the 1D FrFT for 0.5 <= a <= 1.5 (excluding special cases).
    Uses the chirp-multiplication / chirp-convolution / chirp-multiplication method.
    """
    N = len(f)
    alpha = a * np.pi / 2.0
    tana2 = np.tan(alpha / 2.0)
    sina = np.sin(alpha)

    # Centered sample indices: -N/2, ..., N/2-1
    n_c = np.arange(N, dtype=np.float64) - N // 2

    # 1) Pre-chirp multiplication
    pre_chirp = np.exp(-1j * np.pi * tana2 * n_c ** 2 / N)
    g = f * pre_chirp

    # 2) Convolution with chirp kernel h[k] = exp(i*pi*k^2 / (N*sin(alpha)))
    #    k ranges from -(N-1) to (N-1), length 2N-1
    k = np.arange(-(N - 1), N, dtype=np.float64)
    h = np.exp(1j * np.pi * k ** 2 / (N * sina))

    # FFT-based linear convolution (zero-pad to next power of 2 >= 3N)
    Nfft = 1 << int(np.ceil(np.log2(3 * N)))
    conv = np.fft.ifft(np.fft.fft(g, Nfft) * np.fft.fft(h, Nfft))

    # Extract center N samples:
    # offset = (min n_c index in g) negated + (min k index in h) negated
    # = N//2 + (N-1)  => output starts at index N-1
    result = conv[N - 1: 2 * N - 1]

    # 3) Post-chirp multiplication
    result = result * pre_chirp

    # 4) Global scale factor  (includes sample-spacing Δt = 1/√N from the integral)
    result *= np.exp(-1j * np.pi / 4.0 * (1.0 - a)) / np.sqrt(N * np.abs(sina))

    return result


def frft1d(f: np.ndarray, a: float) -> np.ndarray:
    """
    1D Fractional Fourier Transform of order a.

    Parameters
    ----------
    f : 1D numpy array (real or complex)
    a : fractional order
        0  -> identity
        1  -> standard DFT (normalized by 1/sqrt(N))
        2  -> time reversal
        3  -> inverse DFT

    Returns
    -------
    Complex 1D numpy array of the same length.
    """
    f = np.asarray(f, dtype=complex)
    N = len(f)
    a = float(a) % 4.0

    # --- special cases ---
    if a == 0.0:
        return f.copy()
    if a == 2.0:
        return f[::-1].copy()

    sN = np.sqrt(N)
    shft = np.mod(np.arange(N) + N // 2, N).astype(int)

    if a == 1.0:
        out = np.empty(N, dtype=complex)
        out[shft] = np.fft.fft(f[shft]) / sN
        return out
    if a == 3.0:
        out = np.empty(N, dtype=complex)
        out[shft] = np.fft.ifft(f[shft]) * sN
        return out

    # --- reduce to (0.5, 1.5) ---
    if a > 2.0:
        a -= 2.0
        f = f[::-1]
    if a > 1.5:
        a -= 1.0
        out = np.empty(N, dtype=complex)
        out[shft] = np.fft.fft(f[shft]) / sN
        f = out
    if a < 0.5:
        a += 1.0
        out = np.empty(N, dtype=complex)
        out[shft] = np.fft.ifft(f[shft]) * sN
        f = out

    return _frft_core(f, a)


def frft2d(image: np.ndarray, ax: float, ay: float) -> np.ndarray:
    """
    2D Separable Fractional Fourier Transform.

    Applies 1D FrFT of order `ax` along rows (x-axis / columns)
    and 1D FrFT of order `ay` along columns (y-axis / rows).

    Parameters
    ----------
    image : 2D numpy array (H, W), real or complex
    ax    : fractional order along x-axis (columns)
    ay    : fractional order along y-axis (rows)

    Returns
    -------
    Complex 2D array (H, W)
    """
    img = np.asarray(image, dtype=complex)
    H, W = img.shape

    # Apply FrFT along x-axis (each row)
    out = np.empty_like(img)
    for i in range(H):
        out[i, :] = frft1d(img[i, :], ax)

    # Apply FrFT along y-axis (each column)
    result = np.empty_like(out)
    for j in range(W):
        result[:, j] = frft1d(out[:, j], ay)

    return result


def frft2d_spectrum(image: np.ndarray, ax: float, ay: float,
                    log_scale: bool = True) -> np.ndarray:
    """
    Compute the 2D FrFT magnitude spectrum, normalized to [0, 1].

    Parameters
    ----------
    image     : 2D float array (H, W), values in [0, 1]
    ax, ay    : fractional orders for x and y axes
    log_scale : if True, apply log(1 + magnitude) before normalizing

    Returns
    -------
    2D float array (H, W) in [0, 1]
    """
    F = frft2d(image, ax, ay)
    mag = np.abs(F)

    # Shift zero-frequency to center (analogous to fftshift)
    mag = np.fft.fftshift(mag)

    if log_scale:
        mag = np.log1p(mag)

    # Normalize to [0, 1]
    mn, mx = mag.min(), mag.max()
    if mx - mn > 1e-12:
        mag = (mag - mn) / (mx - mn)
    else:
        mag = np.zeros_like(mag)

    return mag
