# ComfyUI DCT Nodes

A ComfyUI custom node pack for 2-D Discrete Cosine Transform (DCT) based frequency-domain filtering.
Draw a mask directly on the DCT spectrum inside the node to zero out any frequency band, then reconstruct the filtered image with the inverse transform.

![이미지 스펙트럼 예시](https://github.com/bemoregt/ComfyUI_DCT/blob/main/ScrShot%207.png)

![이미지 스펙트럼 예시](https://github.com/bemoregt/ComfyUI_DCT/blob/main/ScrShot%208.png)

![이미지 스펙트럼 예시](https://github.com/bemoregt/ComfyUI_DCT/blob/main/ScrShot%209.png)

---

## Nodes

### 1 · DCT 2D Transform
Converts a spatial-domain image into its 2-D DCT-II coefficient map.

| | |
|---|---|
| **Input** | `IMAGE` |
| **Outputs** | `dct_spectrum` — raw DCT coefficients `[B, H, W, C]` (custom `DCT_SPECTRUM` type) |
| | `spectrum_preview` — log-magnitude visualisation normalised to `[0, 1]` (`IMAGE`) |

The transform uses `scipy.fft.dctn` with `norm='ortho'` (orthonormal DCT-II).
Low frequencies sit in the top-left corner; higher frequencies extend toward the bottom-right.

---

### 2 · DCT Spectrum Mask Editor
Applies a user-painted binary mask to the DCT coefficient tensor.
Zeroed coefficients are removed from the spectrum before reconstruction.

| | |
|---|---|
| **Input** | `dct_spectrum` (`DCT_SPECTRUM`) |
| **Outputs** | `masked_spectrum` — filtered DCT coefficients (`DCT_SPECTRUM`) |
| | `mask_preview` — greyscale mask image (`IMAGE`) |

#### Built-in Paint Canvas

The node embeds an interactive paint canvas directly in the ComfyUI graph.
After running the upstream **DCT 2D Transform** node, the spectrum preview is automatically shown as the canvas background.

| Control | Action |
|---------|--------|
| **Left-click / drag** | Paint black → zero out those DCT coefficients |
| **Right-click / drag** | Paint white → restore those coefficients |
| **Scroll wheel** | Increase / decrease brush size (2 – 120 px) |
| **Double-click** | Clear the mask (fill white = keep all frequencies) |

---

### 3 · Inverse DCT 2D
Reconstructs the spatial-domain image from (masked) DCT coefficients.

| | |
|---|---|
| **Input** | `dct_spectrum` (`DCT_SPECTRUM`) |
| **Output** | `reconstructed_image` — recovered image clipped to `[0, 1]` (`IMAGE`) |

Uses `scipy.fft.idctn` with `norm='ortho'`.
Without any mask applied the round-trip error is at machine-epsilon level (< 4 × 10⁻¹⁶).

---

## Recommended Workflow

```
[Load Image]
     │
     ▼
[DCT 2D Transform] ──► spectrum_preview (optional preview node)
     │
     │ dct_spectrum
     ▼
[DCT Spectrum Mask Editor]   ← paint mask on the spectrum canvas
     │
     │ masked_spectrum
     ▼
[Inverse DCT 2D]
     │
     ▼
[Preview Image / Save Image]
```

### Frequency Filtering Examples

| Painted area | Filter type |
|---|---|
| Top-left region (low freq) | High-pass — sharpens edges, removes flat background |
| Outer region (high freq) | Low-pass — blurs image, removes noise and fine detail |
| Custom shape | Arbitrary frequency-domain filter |

---

## Installation

1. Clone or copy this folder into `ComfyUI/custom_nodes/`:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourname/ComfyUI_dct.git
   ```

2. Install the Python dependency:
   ```bash
   pip install scipy
   ```

3. Restart ComfyUI.
   The three nodes appear under the **DCT** category in the node search menu.

### Requirements

| Package | Version |
|---------|---------|
| `scipy` | ≥ 1.4.0 |
| `Pillow` | any recent |
| `torch` | (provided by ComfyUI) |
| `numpy` | (provided by ComfyUI) |

---

## Technical Details

- **Transform**: 2-D DCT Type-II, separable, applied independently per colour channel.
- **Normalisation**: `norm='ortho'` — the forward and inverse transforms are exact inverses of each other.
- **Mask resolution**: The paint canvas uses a 512 × 512 internal raster. At execution time the mask is bilinearly resampled to match the actual spectrum dimensions.
- **Data flow**: `DCT_SPECTRUM` is a standard `torch.Tensor` `[B, H, W, C]` carrying real-valued DCT coefficients; it is not clamped to `[0, 1]`.
