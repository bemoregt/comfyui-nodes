# ComfyUI_FrameInterpolation

A ComfyUI custom node for **2× video frame interpolation** using a Fourier-transform-based phase difference algorithm on a complex steerable pyramid.

---

## Overview

Standard frame interpolation blends pixel intensities linearly between two frames, which produces blurry or ghosted results whenever objects are in motion. This node takes a different approach: instead of blending intensities, it interpolates the **phase** of image content at each spatial scale.

> "Image structure is encoded primarily by phase. Interpolating the phase of each pixel produces a much higher quality result."
> — [Phase-Based Frame Interpolation, exclav.es](https://exclav.es/2016/07/15/phase-based-frame-interpolation-a-julia-implementation/)

The result is a synthesized intermediate frame that preserves sharp edges and fine structure with significantly less ghosting than naive linear blending.

---

## Algorithm

### 1. Color Space Separation

Each input frame is converted from RGB to **CIE Lab** color space. The **L (luminance)** channel carries the structural information processed by phase interpolation. The **a and b (chrominance)** channels are handled with simple linear interpolation, since human vision is far less sensitive to chroma artifacts.

### 2. Complex Analytic Pyramid

A multi-scale complex pyramid is built from the luminance channel of each frame. At every pyramid level, a **2D analytic signal** is computed via FFT by zeroing negative-frequency components (equivalent to a 2D Hilbert transform). This yields complex coefficients where:

- **Amplitude** encodes local contrast at that scale.
- **Phase** encodes the structural position of image features.

### 3. Phase Difference

For each corresponding pyramid level pair `(pyr1[k], pyr2[k])`, the per-pixel phase difference is computed as:

```
Δφ[k] = angle( pyr2[k] · conj(pyr1[k]) )
```

This gives the wrapped phase shift between the two frames at scale `k`, in the range `(−π, π]`.

### 4. Hierarchical Phase Unwrapping

Raw phase differences are ambiguous modulo `2π`. The unwrapping resolves this by propagating motion estimates **coarse-to-fine**:

1. The coarsest level is taken as-is (no ambiguity at low resolution).
2. For each finer level, the coarse unwrapped phase is upsampled and used as a prediction.
3. The residual between the fine difference and the coarse prediction is rounded to the nearest multiple of `2π` and subtracted — effectively choosing the physically consistent branch.

### 5. Phase Consistency Adjustment

A final pass ensures each unwrapped value agrees with the raw phase difference modulo `2π`, correcting any remaining inconsistencies introduced by the upsampling step.

### 6. Interpolation

For an interpolation factor `α ∈ [0, 1]` (midpoint = 0.5):

| Component | Formula |
|-----------|---------|
| Amplitude | `(1 − α) · amp1 + α · amp2` (linear blend) |
| Phase     | `φ1 + α · Δφ_unwrapped` |
| Complex coeff | `amp_interp · exp(i · phase_interp)` |

### 7. Reconstruction

The interpolated complex pyramid is collapsed back into a luminance image by summing real parts across scales. Chrominance channels are linearly blended. Both are then recombined into Lab and converted back to RGB.

### 8. Video Assembly

The output video contains the original frames interleaved with synthesized intermediate frames, doubling the frame count. The output frame rate is set to `2× input FPS` so playback duration is unchanged.

---

## Node Reference

**Category:** `video/interpolation`
**Display Name:** `Phase-Based Frame Interpolation (2x)`
**Internal Name:** `PhaseBasedFrameInterpolation`

### Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_path` | STRING | — | Absolute path to the input video file (mp4, avi, mov, etc.) |
| `pyramid_levels` | INT (3–10) | 6 | Number of pyramid levels. More levels improve unwrapping accuracy at coarser scales but increase processing time. |
| `output_filename` | STRING | `phase_interp_output.mp4` | Filename for the output video, written to ComfyUI's output directory. |

### Outputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_video_path` | STRING | Absolute path to the interpolated output video. |

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone <this-repo-url> ComfyUI_FrameInterpolation
pip install -r ComfyUI_FrameInterpolation/requirements.txt
```

Restart ComfyUI. The node will appear under **video/interpolation**.

---

## Requirements

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations, FFT |
| `scipy` | (available for future extensions) |
| `opencv-python` | Video I/O, color space conversion, pyramid resizing |

---

## Performance Notes

- Processing is **CPU-bound**. A 1-minute 1080p clip at 30 fps may take several minutes depending on hardware.
- `pyramid_levels = 6` balances quality and speed well for most content.
- Increasing `pyramid_levels` improves phase unwrapping for large motions but scales processing time roughly linearly.
- Memory usage scales with the total number of pixels × pyramid levels. Very high-resolution videos should be tested with caution.

---

## Limitations

- The algorithm assumes **small-to-moderate motion** between frames. Large displacements (more than roughly a quarter of the image width per frame) may still produce artifacts, as the phase-unwrapping heuristic can fail at extreme shifts.
- Occlusions and dis-occlusions (objects appearing or disappearing) are not explicitly handled; these regions will blend toward the nearest available content.
- Only luminance is phase-interpolated; chroma is linearly blended, which may produce mild saturation artifacts on color edges under large motion.

---

## References

- [Phase-Based Frame Interpolation — A Julia Implementation](https://exclav.es/2016/07/15/phase-based-frame-interpolation-a-julia-implementation/)
- Simoncelli, E. P., & Freeman, W. T. (1995). *The steerable pyramid: A flexible architecture for multi-scale derivative computation.* ICIP 1995.
- Fleet, D. J., & Jepson, A. D. (1990). *Computation of component image velocity from local phase information.* IJCV 5(1).
