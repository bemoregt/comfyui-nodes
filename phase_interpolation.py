"""
Phase-Based Frame Interpolation using Fourier Transform and Complex Steerable Pyramid

Algorithm reference:
  https://exclav.es/2016/07/15/phase-based-frame-interpolation-a-julia-implementation/

Core concept:
  Image structure is encoded primarily by phase.
  Interpolating phase of each pixel (rather than pixel intensity) produces
  higher quality results with less ghosting.

Steps:
  1. Convert frames to Lab color space, extract luminance (L channel)
  2. Build complex steerable pyramid for each frame
  3. Compute phase differences between corresponding pyramid levels
  4. Unwrap phase differences hierarchically (coarse-to-fine)
  5. Interpolate: amplitude linearly, phase = original + alpha * unwrapped_diff
  6. Reconstruct pyramid -> luminance, blend chrominance linearly
  7. Convert back to RGB
"""

import numpy as np
from scipy.ndimage import zoom
import cv2


# ---------------------------------------------------------------------------
# Utility: color space conversions
# ---------------------------------------------------------------------------

def rgb_to_lab_float(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB uint8 -> Lab float32 using OpenCV float32 convention.

    frame_rgb : (H, W, 3) uint8, values [0, 255]
    returns   : (H, W, 3) float32
                  L  in [0, 100]
                  a  in [-127, 127]
                  b  in [-127, 127]

    Internally OpenCV's float32 Lab path expects BGR input in [0.0, 1.0].
    """
    bgr_f = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
    lab = cv2.cvtColor(bgr_f, cv2.COLOR_BGR2Lab)  # L∈[0,100], a,b∈[-127,127]
    return lab


def lab_to_rgb_uint8(lab: np.ndarray) -> np.ndarray:
    """
    Convert Lab float32 -> RGB uint8.

    lab : (H, W, 3) float32  (L∈[0,100], a,b∈[-127,127])
    returns : (H, W, 3) uint8 [0, 255]
    """
    lab_clipped = lab.copy()
    lab_clipped[:, :, 0] = np.clip(lab_clipped[:, :, 0], 0.0, 100.0)
    lab_clipped[:, :, 1] = np.clip(lab_clipped[:, :, 1], -127.0, 127.0)
    lab_clipped[:, :, 2] = np.clip(lab_clipped[:, :, 2], -127.0, 127.0)
    bgr_f = cv2.cvtColor(lab_clipped, cv2.COLOR_Lab2BGR)   # [0.0, 1.0]
    rgb_f = cv2.cvtColor(bgr_f, cv2.COLOR_BGR2RGB)
    return np.clip(rgb_f * 255.0, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Complex Steerable Pyramid
# ---------------------------------------------------------------------------

def _make_gaussian_kernel(sigma: float = 1.0) -> np.ndarray:
    """Create a small Gaussian kernel for blurring."""
    size = int(6 * sigma + 1) | 1  # ensure odd
    x = np.arange(size) - size // 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def build_laplacian_pyramid(image: np.ndarray, num_levels: int = 6) -> list:
    """
    Build a Laplacian pyramid.

    Each level contains the band-pass detail at that scale.
    The last level is the low-pass residual.

    Returns list of (H_k, W_k) float32 arrays.
    """
    pyramid = []
    current = image.astype(np.float32)
    for _ in range(num_levels - 1):
        blurred = cv2.GaussianBlur(current, (0, 0), sigmaX=1.0)
        h, w = current.shape[:2]
        downsampled = cv2.resize(blurred, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
        upsampled = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_LINEAR)
        laplacian = current - upsampled
        pyramid.append(laplacian)
        current = downsampled
    pyramid.append(current)  # residual
    return pyramid


def build_complex_pyramid(image: np.ndarray, num_levels: int = 6) -> list:
    """
    Build a complex steerable pyramid approximation.

    For each Laplacian level, compute the complex analytic signal using
    the Hilbert transform in 2D (via FFT), giving complex coefficients
    where amplitude encodes local contrast and phase encodes structure.

    Returns list of complex128 arrays, same shape as Laplacian levels.
    """
    lap_pyr = build_laplacian_pyramid(image, num_levels)
    complex_pyr = []
    for band in lap_pyr:
        # 2D analytic signal: zero out negative frequencies
        F = np.fft.fft2(band)
        H, W = band.shape
        # Build half-plane filter (analytic signal in 2D)
        # Use the standard 2D Hilbert / analytic approach:
        # keep DC, double positive half, zero negative half
        mask = np.zeros((H, W), dtype=np.float64)
        # DC component
        mask[0, 0] = 1.0
        # Positive frequencies
        if H % 2 == 0:
            mask[1:H // 2, :] = 2.0
            mask[H // 2, :] = 1.0  # Nyquist row
        else:
            mask[1:(H + 1) // 2, :] = 2.0
        if W % 2 == 0:
            # Already handled via rows; but for W-direction keep symmetric
            pass
        analytic_F = F * mask
        analytic = np.fft.ifft2(analytic_F)
        complex_pyr.append(analytic)
    return complex_pyr


# ---------------------------------------------------------------------------
# Phase operations
# ---------------------------------------------------------------------------

def compute_phase_pyramid(pyr1: list, pyr2: list) -> list:
    """
    Compute phase difference between two complex pyramids.

    phase_diff[k] = angle(pyr2[k] * conj(pyr1[k]))
                  = phase(pyr2[k]) - phase(pyr1[k])

    Returns list of float arrays.
    """
    phase_diffs = []
    for c1, c2 in zip(pyr1, pyr2):
        diff = np.angle(c2 * np.conj(c1))
        phase_diffs.append(diff)
    return phase_diffs


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap values to [-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def unwrap_phase_pyramid(phase_diffs: list) -> list:
    """
    Hierarchical phase unwrapping (coarse to fine).

    For each level k (fine to coarse ordering, i.e. index 0=finest),
    we unwrap by comparing to the next coarser level upsampled back.

    Algorithm:
      1. Start from coarsest level (last index) - no unwrapping needed.
      2. For each finer level k, upsample the unwrapped coarser level,
         compute the residual: diff[k] - wrapped_coarse_up
         round residual to nearest multiple of 2π and subtract.

    This ensures the fine-scale phase delta is consistent with the
    coarse-scale motion estimate.
    """
    num_levels = len(phase_diffs)
    unwrapped = [None] * num_levels

    # Coarsest level: use as-is
    unwrapped[-1] = phase_diffs[-1].copy()

    # Process from coarse to fine
    for k in range(num_levels - 2, -1, -1):
        fine_diff = phase_diffs[k]
        coarse_unwrapped = unwrapped[k + 1]

        # Upsample coarse to fine resolution
        h_fine, w_fine = fine_diff.shape
        coarse_up = cv2.resize(
            coarse_unwrapped.astype(np.float32),
            (w_fine, h_fine),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.float64)

        # Compute residual between fine diff and coarse prediction
        residual = fine_diff - coarse_up
        # Round residual to nearest multiple of 2π
        correction = np.round(residual / (2 * np.pi)) * (2 * np.pi)
        unwrapped[k] = fine_diff - correction

    return unwrapped


def adjust_phase_consistency(phase_diffs: list, unwrapped: list) -> list:
    """
    Final phase consistency adjustment:
    Ensure that (unwrapped mod 2π) == (original diff mod 2π).
    Add/subtract 2π to bring unwrapped as close as possible to original.
    """
    adjusted = []
    for orig, unwr in zip(phase_diffs, unwrapped):
        residual = orig - unwr
        correction = np.round(residual / (2 * np.pi)) * (2 * np.pi)
        adjusted.append(unwr + correction)
    return adjusted


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_from_pyramid(complex_pyr: list) -> np.ndarray:
    """
    Reconstruct image from a complex pyramid by taking real part.

    Reverses the Laplacian pyramid: sum up from coarsest to finest.
    """
    result = np.real(complex_pyr[-1]).astype(np.float32)
    for band in reversed(complex_pyr[:-1]):
        h, w = band.shape
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
        result += np.real(band).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Main interpolation function
# ---------------------------------------------------------------------------

def interpolate_frames(frame1: np.ndarray, frame2: np.ndarray,
                       alpha: float = 0.5, num_levels: int = 6) -> np.ndarray:
    """
    Interpolate between two RGB frames using phase-based method.

    Parameters
    ----------
    frame1, frame2 : (H, W, 3) uint8 RGB
    alpha          : interpolation position [0,1], 0=frame1, 1=frame2
    num_levels     : number of pyramid levels

    Returns
    -------
    interpolated : (H, W, 3) uint8 RGB
    """
    # 1. Convert to Lab, separate L (luminance) from ab (chrominance)
    lab1 = rgb_to_lab_float(frame1)  # (H,W,3)
    lab2 = rgb_to_lab_float(frame2)

    L1 = lab1[:, :, 0] / 100.0   # normalize L [0,100] -> [0,1]
    L2 = lab2[:, :, 0] / 100.0
    ab1 = lab1[:, :, 1:]           # (H,W,2)
    ab2 = lab2[:, :, 1:]

    # 2. Build complex pyramids for luminance
    pyr1 = build_complex_pyramid(L1, num_levels)
    pyr2 = build_complex_pyramid(L2, num_levels)

    # 3. Compute phase differences
    phase_diffs = compute_phase_pyramid(pyr1, pyr2)

    # 4. Unwrap phase differences hierarchically
    unwrapped = unwrap_phase_pyramid(phase_diffs)

    # 5. Adjust phase consistency
    adjusted = adjust_phase_consistency(phase_diffs, unwrapped)

    # 6. Build interpolated pyramid
    interp_pyr = []
    for (c1, c2, adj_diff) in zip(pyr1, pyr2, adjusted):
        # Linearly interpolate amplitude
        amp1 = np.abs(c1)
        amp2 = np.abs(c2)
        amp_interp = (1 - alpha) * amp1 + alpha * amp2

        # Phase of frame1 + scaled phase difference
        phase1 = np.angle(c1)
        phase_interp = phase1 + alpha * adj_diff

        # Reconstruct complex coefficients
        c_interp = amp_interp * np.exp(1j * phase_interp)
        interp_pyr.append(c_interp)

    # 7. Reconstruct luminance
    L_interp = reconstruct_from_pyramid(interp_pyr)
    # Clamp and rescale back to [0, 100] Lab L range
    L_interp = np.clip(L_interp * 100.0, 0, 100)

    # 8. Linearly interpolate chrominance (ab channels)
    ab_interp = (1 - alpha) * ab1 + alpha * ab2

    # 9. Assemble Lab and convert back to RGB
    lab_interp = np.zeros_like(lab1)
    lab_interp[:, :, 0] = L_interp
    lab_interp[:, :, 1:] = ab_interp

    rgb_interp = lab_to_rgb_uint8(lab_interp)
    return rgb_interp


# ---------------------------------------------------------------------------
# Batch (tensor-based) interpolation  — used by the ComfyUI node
# ---------------------------------------------------------------------------

def interpolate_batch_2x(frames: list, num_levels: int = 6) -> list:
    """
    Insert one interpolated frame between every consecutive pair.

    Parameters
    ----------
    frames     : list of (H, W, 3) uint8 RGB arrays
    num_levels : steerable pyramid levels

    Returns
    -------
    out_frames : list length 2*N-1, alternating originals and interpolated
    """
    if len(frames) < 2:
        return frames

    out = []
    for i, frame in enumerate(frames):
        out.append(frame)
        if i < len(frames) - 1:
            interp = interpolate_frames(frame, frames[i + 1],
                                        alpha=0.5, num_levels=num_levels)
            out.append(interp)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"[PhaseInterp] {i + 1}/{len(frames) - 1} pairs done")
    return out


# ---------------------------------------------------------------------------
# Video file processing  (kept for standalone / CLI use)
# ---------------------------------------------------------------------------

def interpolate_video_2x(input_path: str, output_path: str,
                          num_levels: int = 6) -> str:
    """
    Load video, insert one interpolated frame between each pair of frames (2x).

    Parameters
    ----------
    input_path  : path to input video
    output_path : path to output video
    num_levels  : steerable pyramid levels

    Returns
    -------
    output_path : same as output_path on success
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output fps is 2x (more frames, same duration)
    out_fps = fps * 2.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    # Read all frames into memory (for moderate-length clips)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) < 2:
        raise RuntimeError("Video must have at least 2 frames")

    print(f"[PhaseInterp] Input: {len(frames)} frames @ {fps:.2f} fps")
    print(f"[PhaseInterp] Output: {len(frames) * 2 - 1} frames @ {out_fps:.2f} fps")

    for i, frame in enumerate(frames):
        # Write original frame
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Write interpolated frame between current and next
        if i < len(frames) - 1:
            interp = interpolate_frames(frame, frames[i + 1],
                                        alpha=0.5, num_levels=num_levels)
            writer.write(cv2.cvtColor(interp, cv2.COLOR_RGB2BGR))
            if (i + 1) % 10 == 0 or i == 0:
                print(f"[PhaseInterp] Processed {i + 1}/{len(frames) - 1} frame pairs")

    writer.release()
    print(f"[PhaseInterp] Done -> {output_path}")
    return output_path
