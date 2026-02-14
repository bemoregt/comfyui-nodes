# ComfyUI Spectral Residual Saliency

A ComfyUI custom node that computes a **Spectral Residual Saliency Map** from an input image.

Based on the paper:
> Hou, X., & Zhang, L. (2007). *Saliency Detection: A Spectral Residual Approach.* CVPR 2007.

---

## Algorithm

The spectral residual method detects visually salient regions by finding the irregular (non-redundant) parts of the image spectrum:

1. Convert the input image to grayscale.
2. Apply 2D FFT and compute the log amplitude spectrum: `A(f) = log|F(f)|`
3. Estimate the average spectrum using a uniform filter: `Ā(f) = h_n * A(f)`
4. Compute the spectral residual: `R(f) = A(f) - Ā(f)`
5. Reconstruct the saliency map via IFFT: `S(x) = |IFFT(exp(R(f) + iΦ(f)))|²`
6. Apply Gaussian smoothing to the result.

The residual `R(f)` captures the parts of the spectrum that deviate from the statistical average — these correspond to visually conspicuous regions in the image.

---

## Installation

Copy or symlink this folder into your ComfyUI `custom_nodes` directory:

```bash
cp -r ComfyUI_srsm /path/to/ComfyUI/custom_nodes/
```

### Dependencies

The node requires `scipy` in addition to the standard ComfyUI environment:

```bash
pip install scipy
```

---

## Node

**Category:** `image/analysis`
**Display name:** `Image to Spectral Residual Saliency`

### Inputs

| Name | Type | Description |
|---|---|---|
| `image` | `IMAGE` | Input image (batch supported) |

### Parameters

| Name | Default | Range | Description |
|---|---|---|---|
| `colormap` | `hot` | — | Color palette applied to the saliency map (`hot`, `inferno`, `magma`, `jet`, `viridis`, `plasma`, `gray`) |
| `smooth_sigma` | `8.0` | 0 – 32 | Gaussian smoothing σ applied after IFFT. Set to `0` to disable. |
| `avg_filter_size` | `3` | 1 – 15 | Size of the uniform filter used to estimate the average log spectrum. Odd values recommended. |
| `output_mode` | `colormap` | — | Output format: `colormap` (colored saliency), `grayscale` (normalized grayscale), or `overlay` (saliency blended over the original image) |
| `overlay_alpha` | `0.6` | 0 – 1 | Opacity of the saliency layer in `overlay` mode. `0` = original only, `1` = saliency only. |

### Output

| Name | Type | Description |
|---|---|---|
| `saliency_map` | `IMAGE` | Computed saliency map, same spatial resolution as the input |

---

## Output Modes

| Mode | Description |
|---|---|
| `colormap` | Saliency values mapped to a color palette. Good for visualization and inspection. |
| `grayscale` | Normalized saliency as a single-channel grayscale image (replicated to RGB). |
| `overlay` | Colorized saliency blended on top of the original image, controlled by `overlay_alpha`. |

---

## Notes

- Batch inputs are fully supported; each image in the batch is processed independently.
- The node is CPU-based. For large batches or high-resolution images, processing time scales with batch size.
- `avg_filter_size` corresponds to the parameter `n` in the original paper (default `n=3`).
- `smooth_sigma` corresponds to the Gaussian kernel described in the paper (recommended σ ≈ 8).
