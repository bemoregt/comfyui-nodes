# ComfyUI QFT Spectral Residual Saliency Map

A ComfyUI custom node that computes a **visual saliency map** from a color image using the **Quaternion Fourier Transform (QFT)** combined with the **Spectral Residual (SR)** method.

Unlike conventional per-channel approaches, this node encodes all three RGB channels simultaneously as a single pure quaternion, preserving cross-channel color correlations throughout the frequency analysis.

---

## How It Works

The algorithm follows the QFT-based spectral residual approach described by Schauerte & Stiefelhagen (2012):

### 1. Pure Quaternion Encoding

Each pixel `(R, G, B)` is encoded as a pure quaternion:

```
q(x, y) = R·i + G·j + B·k
```

The scalar part is zero, and each color channel occupies one of the three imaginary axes.

### 2. Quaternion Fourier Transform (QFT)

The left-sided QFT with axis **μ = i** is applied. Using the **symplectic decomposition**, any pure quaternion `q` can be split into two complex arrays:

```
q = p + q_c · j
  where  p   = R·i       (complex: imaginary part = R)
         q_c = G + B·i   (complex: real = G, imaginary = B)
```

This allows the QFT to be computed via exactly **two standard 2-D complex FFTs**:

```
F_L{q}(u, v) = DFT(p)(u, v) + DFT(q_c)(u, v) · j
```

### 3. Log Amplitude & Spectral Residual

The quaternion magnitude at each frequency bin is:

```
A(u, v) = log( sqrt( |DFT(p)|² + |DFT(q_c)|² ) )
```

A box (average) filter of size `smooth_kernel` is applied to obtain a smooth "prior" spectrum:

```
Ā(u, v) = h_n * A(u, v)
```

The **spectral residual** is the deviation from this prior:

```
SR(u, v) = A(u, v) - Ā(u, v)
```

Visually distinct (salient) regions produce unusually large spectral residuals because they deviate from the repetitive, average structure of the scene.

### 4. Inverse QFT & Saliency Map

The residual spectrum is reconstructed by applying the weight `exp(-Ā)` to the original QFT:

```
F_SR = exp(-Ā) · F_L{q}   =   exp(SR) · unit_quaternion
```

The inverse QFT is computed via two inverse FFTs, and the saliency at each pixel is the squared quaternion magnitude:

```
S(x, y) = |IQFT{F_SR}(x, y)|²
         = |IDFT(P_sr)|² + |IDFT(Q_sr)|²
```

Finally, `S` is smoothed with a Gaussian filter (`gaussian_sigma`) and normalized to `[0, 1]`.

---

## Node Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `image` | IMAGE | — | — | Input color image (batches supported) |
| `smooth_kernel` | INT | `8` | 1 – 128 | Size of the average filter on the log spectrum. Larger values produce higher contrast between salient and non-salient regions. |
| `gaussian_sigma` | FLOAT | `8.0` | 0.0 – 100.0 | Sigma of the Gaussian blur applied to the final saliency map. Larger values produce smoother, more blob-like saliency regions. |
| `output_mode` | ENUM | `grayscale` | grayscale / heatmap | `grayscale` — white-on-black intensity map. `heatmap` — jet colormap (blue → green → red). |

**Output:** `saliency_map` — IMAGE of the same spatial resolution as the input, float32 in `[0, 1]`.

---

## Installation

1. Clone or copy this folder into your ComfyUI `custom_nodes` directory:

   ```bash
   git clone <repo-url> /path/to/ComfyUI/custom_nodes/ComfyUI_QFT_SRSM
   ```

   or copy manually:

   ```bash
   cp -r ComfyUI_QFT_SRSM /path/to/ComfyUI/custom_nodes/
   ```

2. Install the optional (but recommended) dependency:

   ```bash
   pip install scipy
   ```

   If `scipy` is not available, the node automatically falls back to a pure-NumPy implementation of the box and Gaussian filters.

3. Restart ComfyUI. The node will appear under **`image/analysis`** as **"QFT Spectral Residual Saliency"**.

---

## Dependencies

| Package | Required | Notes |
|---------|----------|-------|
| `numpy` | Yes | Bundled with ComfyUI |
| `torch` | Yes | Bundled with ComfyUI |
| `scipy` | Recommended | Falls back to NumPy if missing |

---

## Usage Tips

- **`smooth_kernel`** controls how much of the "average" scene statistics are subtracted. A value of `4`–`16` works well for most images. Very small values (1–2) may produce noisy results; very large values may suppress fine-grained salient details.
- **`gaussian_sigma`** should be scaled with image resolution. For a 512×512 image, values of `8`–`16` are typical starting points.
- The `heatmap` output mode is useful for visual inspection; connect it directly to a **Preview Image** node. Use `grayscale` mode when feeding the saliency map into downstream masking or compositing nodes.
- The node processes each image in the batch independently and supports any resolution.

---

## References

- Hou, X. & Zhang, L. (2007). **Saliency Detection: A Spectral Residual Approach.** *CVPR 2007.*
- Schauerte, B. & Stiefelhagen, R. (2012). **Quaternion-based Spectral Saliency Detection for Eye Fixation Prediction.** *ECCV 2012.*
- Ell, T. A. & Sangwine, S. J. (2007). **Hypercomplex Fourier Transforms of Color Images.** *IEEE Transactions on Image Processing.*
