"""
QFT Spectral Residual Saliency Map - ComfyUI Custom Node

Computes a saliency map using Quaternion Fourier Transform (QFT) and the
Spectral Residual (SR) method for color images.

Algorithm (Schauerte & Stiefelhagen, 2012):
  1. Encode RGB as pure quaternion: q = R*i + G*j + B*k
  2. Left-sided QFT with axis mu=i via symplectic decomposition:
       decompose q = p + q_c*j  where p = R*i, q_c = G + B*i  (complex)
       F_L{q}(u,v) = DFT(p)(u,v) + DFT(q_c)(u,v) * j
  3. Log amplitude: A(u,v) = log(|F_L{q}(u,v)|)
  4. Smooth A with average filter: A_bar = h_n * A
  5. Spectral residual weight: w = exp(-A_bar)
  6. Residual spectrum: F_SR = w * F_L{q}
  7. Inverse QFT: f_SR = IQFT{F_SR}  (two IFFTs)
  8. Saliency: S = |f_SR|^2, smoothed with Gaussian
"""

import numpy as np
import torch

try:
    from scipy.ndimage import gaussian_filter, uniform_filter
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ---------------------------------------------------------------------------
# Fallback filters (numpy-only, used when scipy is not available)
# ---------------------------------------------------------------------------

def _uniform_filter_np(arr: np.ndarray, size: int) -> np.ndarray:
    """Box (average) filter using integral image / cumulative sum."""
    H, W = arr.shape
    pad = size // 2
    arr_p = np.pad(arr, pad, mode="edge")
    # Use 2-D sliding window via cumsum trick
    # Cumsum along rows
    cs = np.cumsum(arr_p, axis=0)
    cs = cs[size:, :] - cs[:-size, :]
    # Cumsum along cols
    cs = np.cumsum(cs, axis=1)
    cs = cs[:, size:] - cs[:, :-size]
    return cs / (size * size)


def _gaussian_filter_np(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian filter via multiplication in frequency domain."""
    H, W = arr.shape
    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    FX, FY = np.meshgrid(fx, fy)
    kernel_fft = np.exp(-2.0 * np.pi ** 2 * sigma ** 2 * (FX ** 2 + FY ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(arr) * kernel_fft))


def _uf(arr, size):
    if _SCIPY:
        return uniform_filter(arr, size=size)
    return _uniform_filter_np(arr, size)


def _gf(arr, sigma):
    if _SCIPY:
        return gaussian_filter(arr, sigma=sigma)
    return _gaussian_filter_np(arr, sigma)


# ---------------------------------------------------------------------------
# Jet colormap (no matplotlib dependency)
# ---------------------------------------------------------------------------

def _jet_colormap(gray: np.ndarray) -> np.ndarray:
    """Map a [0,1] grayscale array to RGB using a jet-like colormap."""
    r = np.clip(1.5 - np.abs(4.0 * gray - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * gray - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * gray - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# Core QFT-SR saliency computation
# ---------------------------------------------------------------------------

def qft_spectral_residual(
    R: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
    smooth_kernel: int,
    gaussian_sigma: float,
) -> np.ndarray:
    """
    Compute the QFT-based Spectral Residual saliency map.

    Parameters
    ----------
    R, G, B : (H, W) float64 arrays in [0, 1]
    smooth_kernel : averaging kernel size (odd)
    gaussian_sigma : Gaussian smoothing sigma for final map

    Returns
    -------
    saliency : (H, W) float32 in [0, 1]
    """
    # --- Step 1: Symplectic decomposition ---
    # q = R*i + G*j + B*k  =>  q = p + q_c*j
    #   p   = 0 + R*i   (complex, imaginary part = R)
    #   q_c = G + B*i   (complex, real=G, imag=B)
    p_complex = R * 1j          # shape (H, W), dtype complex128
    q_complex = G + B * 1j     # shape (H, W), dtype complex128

    # --- Step 2: Left QFT via two complex DFTs ---
    P = np.fft.fft2(p_complex)     # 'i-part' of QFT result
    Q = np.fft.fft2(q_complex)     # 'j-part' coefficient of QFT result

    # --- Step 3: Quaternion magnitude ---
    magnitude = np.sqrt(np.abs(P) ** 2 + np.abs(Q) ** 2)
    magnitude = np.maximum(magnitude, 1e-12)

    # --- Step 4: Log amplitude ---
    log_amp = np.log(magnitude)

    # --- Step 5: Smooth log amplitude ---
    k = smooth_kernel
    k = k if k % 2 == 1 else k + 1   # ensure odd
    k = max(k, 1)
    log_amp_smooth = _uf(log_amp, k)

    # --- Step 6: Spectral residual weight ---
    # SR(u,v) = log_amp - log_amp_smooth
    # Residual spectrum: exp(SR) * unit_quaternion = exp(-log_amp_smooth) * QFT
    weight = np.exp(-log_amp_smooth)   # shape (H, W), real

    P_sr = weight * P
    Q_sr = weight * Q

    # --- Step 7: Inverse QFT ---
    # IQFT{P_sr + Q_sr*j} = IDFT(P_sr) + IDFT(Q_sr)*j
    p_rec = np.fft.ifft2(P_sr)
    q_rec = np.fft.ifft2(Q_sr)

    # --- Step 8: Saliency = |quaternion|^2 ---
    saliency = np.abs(p_rec) ** 2 + np.abs(q_rec) ** 2

    # --- Step 9: Gaussian smoothing ---
    if gaussian_sigma > 0.0:
        saliency = _gf(saliency, gaussian_sigma)

    # --- Step 10: Normalize to [0, 1] ---
    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min:
        saliency = (saliency - s_min) / (s_max - s_min)
    else:
        saliency = np.zeros_like(saliency)

    return saliency.astype(np.float32)


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class QFTSpectralResidualSaliency:
    """
    QFT Spectral Residual Saliency Map

    Generates a visual saliency map from a color image using the
    Quaternion Fourier Transform (QFT) and Spectral Residual (SR) method.

    The three RGB channels are encoded as the pure-quaternion components
    (i, j, k) so that cross-channel correlations are captured holistically
    rather than independently per channel.

    Inputs
    ------
    image            : Input color image (batch supported)
    smooth_kernel    : Size of the average filter applied to the log spectrum
                       (larger = more contrast in saliency)
    gaussian_sigma   : Gaussian blur sigma for the final saliency map
                       (larger = smoother saliency blobs)
    output_mode      : "grayscale" outputs a white-on-black map;
                       "heatmap" applies a jet colormap (blue→red)

    Output
    ------
    saliency_map : IMAGE in [0, 1], same spatial size as input
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "smooth_kernel": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "gaussian_sigma": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "display": "number",
                    },
                ),
                "output_mode": (
                    ["grayscale", "heatmap"],
                    {"default": "grayscale"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("saliency_map",)
    FUNCTION = "compute"
    CATEGORY = "image/analysis"

    def compute(
        self,
        image: torch.Tensor,
        smooth_kernel: int = 8,
        gaussian_sigma: float = 8.0,
        output_mode: str = "grayscale",
    ):
        """
        Parameters
        ----------
        image : torch.Tensor  shape (B, H, W, C), float32 in [0, 1]

        Returns
        -------
        tuple[torch.Tensor]  shape (B, H, W, 3), float32 in [0, 1]
        """
        batch_out = []

        for b in range(image.shape[0]):
            img = image[b].cpu().numpy().astype(np.float64)   # (H, W, C)
            H, W, C = img.shape

            # Extract R, G, B (fall back gracefully for single-channel inputs)
            if C >= 3:
                R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            else:
                lum = img[:, :, 0]
                R, G, B = lum, lum, lum

            sal = qft_spectral_residual(R, G, B, smooth_kernel, gaussian_sigma)

            if output_mode == "heatmap":
                out = _jet_colormap(sal).astype(np.float32)   # (H, W, 3)
            else:
                out = np.stack([sal, sal, sal], axis=-1)       # (H, W, 3)

            batch_out.append(out)

        result = np.stack(batch_out, axis=0)   # (B, H, W, 3)
        return (torch.from_numpy(result),)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "QFTSpectralResidualSaliency": QFTSpectralResidualSaliency,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QFTSpectralResidualSaliency": "QFT Spectral Residual Saliency",
}
