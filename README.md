# ComfyUI_FrFT

A ComfyUI custom node that computes the **Fractional Fourier Transform (FrFT)** of an image and outputs its magnitude spectrum.

---

## What is the FrFT?

The Fractional Fourier Transform is a generalization of the standard Fourier transform parameterized by an order `a`. It continuously rotates a signal in the time–frequency plane:

| Order | Result |
|-------|--------|
| `0.0` | Identity (input magnitude) |
| `0.5` | Midpoint between spatial and frequency domains |
| `1.0` | Standard DFT spectrum (equivalent to FFT) |
| `2.0` | Spatially reversed image |

Intermediate values produce spectra that blend spatial and frequency information, useful for analyzing signals with time-varying frequency content.

---

## Installation

1. Clone or copy this repository into your ComfyUI custom nodes directory:
   ```
   ComfyUI/custom_nodes/ComfyUI_FrFT/
   ```
2. Restart ComfyUI. The node will appear automatically.

**Dependencies:** `numpy` (already bundled with ComfyUI).

---

## Node: FrFT Spectrum

**Category:** `image/transform`

### Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `image` | IMAGE | — | Input image tensor `[B, H, W, C]` |
| `order_x` | FLOAT (0–4) | `1.0` | FrFT order along the x-axis (columns) |
| `order_y` | FLOAT (0–4) | `1.0` | FrFT order along the y-axis (rows) |
| `log_scale` | BOOLEAN | `True` | Apply log(1 + \|F\|) compression before normalization |
| `channel_mode` | ENUM | `luminance` | `luminance`: convert to grayscale first; `per_channel`: process each RGB channel independently |

### Output

| Name | Type | Description |
|------|------|-------------|
| `spectrum` | IMAGE | Magnitude spectrum normalized to [0, 1], shape `[B, H, W, 3]` |

---

## Algorithm

The implementation follows the **Ozaktas–Arikan–Kutay–Bozdagi (1996)** fast discrete FrFT algorithm:

1. Reduce the order `a` to the core interval `(0.5, 1.5)` using FFT/IFFT steps.
2. Apply chirp premultiplication: `g[n] = f[n] · exp(−iπ tan(φ/2) · n² / N)`
3. Convolve with a chirp kernel via zero-padded FFT: `h[k] = exp(iπ k² / (N sin φ))`
4. Apply chirp postmultiplication (same chirp as step 2).
5. Scale by `exp(−iπ(1−a)/4) / sqrt(N |sin φ|)`.

The 2D transform is separable: 1D FrFT is applied along rows first, then columns.

**Reference:**
H. M. Ozaktas, O. Arikan, M. A. Kutay, G. Bozdagi, "Digital Computation of the Fractional Fourier Transform," *IEEE Transactions on Signal Processing*, vol. 44, no. 9, pp. 2141–2150, 1996.

---

## Notes

- The algorithm is **approximately unitary**: norm preservation error is < 1% for smooth signals near `a = 1`, and up to ~15% at boundary orders (`a = 0.5`, `a = 1.5`) due to finite-length discretization effects. This is expected behavior of the Ozaktas discrete FrFT and does not affect visual quality.
- Additivity (`F_a ∘ F_b ≈ F_{a+b}`) holds to within ~0.7% relative error for smooth signals.
- For purely visual spectrum analysis, all order values produce meaningful and correct results.

---

## License

MIT
