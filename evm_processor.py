"""
Eulerian Video Magnification core processor.

Based on: Wu et al., "Eulerian Video Magnification for Revealing Subtle Changes in the World"
SIGGRAPH 2012.

Key idea (Eulerian perspective):
  Instead of tracking where each pixel moves (Lagrangian / optical flow),
  we fix our observation at each spatial location and analyze how its intensity
  oscillates over time. Bandpass filtering isolates the frequency range of
  interest, and the amplified signal is added back onto the original — making
  invisible micro-motions visible.
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy import signal as scipy_signal


# ──────────────────────────────────────────────────────────────────────────────
# Color space helpers
# ──────────────────────────────────────────────────────────────────────────────

_RGB2YIQ = np.array([
    [0.299,  0.587,  0.114],
    [0.596, -0.274, -0.322],
    [0.211, -0.523,  0.312],
], dtype=np.float64)

# Exact inverse — guarantees lossless round-trip
_YIQ2RGB = np.linalg.inv(_RGB2YIQ)


def rgb2yiq(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) float32 RGB [0,1] → YIQ."""
    return (img.astype(np.float64) @ _RGB2YIQ.T).astype(np.float32)


def yiq2rgb(img: np.ndarray) -> np.ndarray:
    """(H, W, 3) float32 YIQ → RGB."""
    return (img.astype(np.float64) @ _YIQ2RGB.T).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Spatial pyramid
# ──────────────────────────────────────────────────────────────────────────────

def laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """
    Build a Laplacian pyramid for a single float32 image.
    Returns list of length `levels`:
      pyramid[0] ... pyramid[-2]  : Laplacian bands (detail)
      pyramid[-1]                 : coarsest Gaussian residual
    """
    gauss = [img.astype(np.float32)]
    for _ in range(levels - 1):
        gauss.append(cv2.pyrDown(gauss[-1]))

    lap = []
    for i in range(levels - 1):
        h, w = gauss[i].shape[:2]
        up = cv2.pyrUp(gauss[i + 1], dstsize=(w, h))
        lap.append(gauss[i] - up)
    lap.append(gauss[-1])
    return lap


def reconstruct_from_laplacian(pyramid: list[np.ndarray]) -> np.ndarray:
    """Collapse a Laplacian pyramid back to a full-resolution image."""
    img = pyramid[-1].copy()
    for i in range(len(pyramid) - 2, -1, -1):
        h, w = pyramid[i].shape[:2]
        img = cv2.pyrUp(img, dstsize=(w, h))
        img = img + pyramid[i]
    return img


def build_video_pyramid(
    frames: np.ndarray, levels: int
) -> list[np.ndarray]:
    """
    Build Laplacian pyramid for every frame and reorganize by level.

    Args:
        frames : (T, H, W, C) float32
        levels : number of pyramid levels

    Returns:
        list of length `levels`, each element is (T, h_l, w_l, C)
    """
    T = len(frames)
    by_level: list[list] = [[] for _ in range(levels)]

    for t in range(T):
        pyr = laplacian_pyramid(frames[t], levels)
        for l in range(levels):
            by_level[l].append(pyr[l])

    return [np.stack(lv, axis=0) for lv in by_level]


# ──────────────────────────────────────────────────────────────────────────────
# Temporal filtering
# ──────────────────────────────────────────────────────────────────────────────

def design_bandpass(
    low_hz: float, high_hz: float, fps: float, order: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Return (b, a) Butterworth bandpass filter coefficients."""
    nyq = fps / 2.0
    lo = float(np.clip(low_hz / nyq, 1e-6, 1.0 - 1e-6))
    hi = float(np.clip(high_hz / nyq, 1e-6, 1.0 - 1e-6))
    if lo >= hi:
        hi = min(lo + 0.05, 1.0 - 1e-6)
    return scipy_signal.butter(order, [lo, hi], btype="band")


def temporal_filter(stack: np.ndarray, b, a) -> np.ndarray:
    """
    Zero-phase bandpass filter along time axis (axis 0).

    Args:
        stack : (T, ...) float32
        b, a  : filter coefficients from design_bandpass()

    Returns:
        filtered array, same shape as stack
    """
    T = stack.shape[0]
    flat = stack.reshape(T, -1)
    out = scipy_signal.filtfilt(b, a, flat, axis=0)
    return out.reshape(stack.shape).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Main EVM pipeline
# ──────────────────────────────────────────────────────────────────────────────

class EVMProcessor:
    """Eulerian Video Magnification processor."""

    def process(
        self,
        frames: np.ndarray,
        fps: float,
        alpha: float,
        low_hz: float,
        high_hz: float,
        pyramid_levels: int = 4,
        filter_order: int = 1,
        amplify_levels: list[int] | None = None,
        color_space: str = "rgb",
    ) -> np.ndarray:
        """
        Amplify subtle temporal variations in a video clip.

        Args:
            frames         : (T, H, W, C) float32, values in [0, 1]
            fps            : frame rate of the video
            alpha          : amplification factor
            low_hz         : bandpass lower cutoff (Hz)
            high_hz        : bandpass upper cutoff (Hz)
            pyramid_levels : Laplacian pyramid depth
            filter_order   : Butterworth filter order (1 = gentle, 5 = steep)
            amplify_levels : which pyramid levels to amplify
                             (None = all levels)
            color_space    : "rgb" or "yiq"
                             YIQ lets you separate luma motion from chroma

        Returns:
            (T, H, W, C) float32 array clipped to [0, 1]
        """
        T, H, W, C = frames.shape
        print(f"[EVM] {T} frames  {W}×{H}  {fps:.1f} fps")
        print(f"[EVM] α={alpha}  bandpass [{low_hz:.3f}, {high_hz:.3f}] Hz"
              f"  levels={pyramid_levels}  order={filter_order}")

        # ── 1. Color space conversion ─────────────────────────────────────────
        if color_space == "yiq":
            work = np.stack([rgb2yiq(frames[t]) for t in range(T)])
        else:
            work = frames.copy()

        # ── 2. Spatial decomposition ──────────────────────────────────────────
        print("[EVM] Building Laplacian pyramids …")
        pyr_by_level = build_video_pyramid(work, pyramid_levels)

        # ── 3. Temporal bandpass + amplification ──────────────────────────────
        print("[EVM] Temporal filtering …")
        b, a = design_bandpass(low_hz, high_hz, fps, filter_order)

        if amplify_levels is None:
            amplify_levels = list(range(pyramid_levels))

        amp_by_level = []
        for l, level_stack in enumerate(pyr_by_level):
            if l in amplify_levels:
                filtered = temporal_filter(level_stack, b, a)
                amp_by_level.append(filtered * alpha)
            else:
                amp_by_level.append(np.zeros_like(level_stack))

        # ── 4. Reconstruction ─────────────────────────────────────────────────
        print("[EVM] Reconstructing frames …")
        result = np.empty_like(work)
        for t in range(T):
            delta_pyr = [amp_by_level[l][t] for l in range(pyramid_levels)]
            delta = reconstruct_from_laplacian(delta_pyr)
            result[t] = work[t] + delta

        # ── 5. Color space back ───────────────────────────────────────────────
        if color_space == "yiq":
            result = np.stack([yiq2rgb(result[t]) for t in range(T)])

        return np.clip(result, 0.0, 1.0).astype(np.float32)
