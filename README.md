# ComfyUI 3DGS Preview Node

A ComfyUI custom node that takes **two images** as input and outputs a **novel-view preview rendered with 3D Gaussian Splatting**.

---

## Overview

This node reconstructs a sparse 3D scene from a pair of photographs and renders it from a user-defined viewpoint using Gaussian Splatting. No pre-trained model is required — everything runs from scratch using classical computer vision (feature matching + triangulation) followed by a differentiable Gaussian renderer.

```
[Image 1] ──┐
             ├──► Feature Matching ──► Camera Pose ──► Triangulation ──► Gaussian Render ──► [Preview]
[Image 2] ──┘
```

> **Note:** Full 3DGS training typically requires dozens to hundreds of images. With only two images the scene is reconstructed as a **sparse point cloud**; the output is an informative structural preview, not a dense photorealistic render.

---

## Installation

### 1. Place the node in ComfyUI

```bash
# From your ComfyUI root directory
cd custom_nodes
git clone <this-repo-url> ComfyUI_3DGS
# — or — copy the folder manually into custom_nodes/
```

### 2. Install dependencies

```bash
pip install -r custom_nodes/ComfyUI_3DGS/requirements.txt
```

### 3. Restart ComfyUI

After restart, search for **"3DGS Preview"** in the node browser (under `3D / Gaussian Splatting`).

---

## Requirements

| Package | Minimum Version |
|---------|----------------|
| `torch` | 2.0.0 |
| `opencv-python` | 4.7.0 |
| `numpy` | 1.24.0 |
| `Pillow` | 9.0.0 |

GPU (CUDA) is used automatically when available; CPU fallback is supported.

---

## Node Reference

**Category:** `3D / Gaussian Splatting`
**Node name:** `GaussianSplattingPreview`

### Inputs

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `image1` | IMAGE | Yes | First input image |
| `image2` | IMAGE | Yes | Second input image (different viewpoint of the same scene) |
| `render_width` | INT | No | Output image width in pixels (default: 512) |
| `render_height` | INT | No | Output image height in pixels (default: 512) |
| `max_gaussians` | INT | No | Maximum number of Gaussians (= triangulated points). Range: 100–5000, default: 1000 |
| `gaussian_scale` | FLOAT | No | Size of each Gaussian relative to focal length. Larger = softer/bigger blobs. Range: 0.001–0.3, default: 0.025 |
| `novel_view_yaw` | FLOAT | No | Horizontal rotation of the novel viewpoint in degrees. 0 = front, +/− = left/right. Range: −90–90, default: 15 |
| `novel_view_pitch` | FLOAT | No | Vertical tilt of the novel viewpoint in degrees. Range: −45–45, default: −5 |
| `background_color` | ENUM | No | Background fill: `dark` / `white` / `black` / `transparent` (default: `dark`) |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `rendered_preview` | IMAGE | Novel-view image rendered with Gaussian Splatting |

---

## How It Works

### Step 1 — Feature Detection & Matching
SIFT keypoints are detected in both images (falls back to ORB if SIFT is unavailable). Matches are filtered with Lowe's ratio test (`threshold = 0.75`).

### Step 2 — Camera Pose Estimation
The Essential Matrix is computed via RANSAC, then decomposed into rotation **R** and translation **t** between the two cameras.

### Step 3 — Point Triangulation
Matched feature pairs are triangulated using the two projection matrices, yielding a sparse 3D point cloud. Points are filtered for validity (finite values, positive depth, depth-outlier removal) and normalized to unit scale.

### Step 4 — Gaussian Initialization
Each triangulated 3D point becomes one isotropic Gaussian:
- **Position** — triangulated 3D coordinate
- **Color** — sampled from `image1` at the feature pixel location
- **Scale (sigma)** — derived from `gaussian_scale` and depth at render time

### Step 5 — Novel-View Rendering
A virtual camera is placed at `z = 2.5` (normalized scene units) and rotated by the specified yaw/pitch angles. Gaussians are:
1. Projected onto the screen using a pinhole camera model
2. Sorted back-to-front (painter's algorithm)
3. Rendered using batched weighted-additive splatting on PyTorch tensors
4. Composited over the chosen background using accumulated opacity

```
alpha(h,w) = 1 − exp(−weight(h,w) × 2.5)
rgb(h,w)   = (Σ gauss_i · color_i) / (Σ gauss_i)
out(h,w)   = rgb · alpha + background · (1 − alpha)
```

### Fallback Behavior
If reconstruction fails (too few keypoints, poor matches, degenerate pose), the node falls back to a **side-by-side** display of the two input images instead of returning an error.

---

## Tips for Best Results

| Situation | Recommendation |
|-----------|----------------|
| Blank or nearly blank output | Increase `novel_view_yaw` closer to 0, or check that images have enough texture |
| Too sparse / many holes | Increase `max_gaussians` (up to 5000) or use higher-resolution inputs |
| Blobs too large | Decrease `gaussian_scale` (try 0.01–0.015) |
| Blobs too small / invisible | Increase `gaussian_scale` (try 0.04–0.08) |
| Reconstruction always fails | Ensure the two images show the **same scene** from **slightly different viewpoints** (10–40° apart works best) |
| Flat / textureless images | SIFT/ORB need texture to find keypoints; add some surface detail |

---

## File Structure

```
ComfyUI_3DGS/
├── __init__.py          # Node registration for ComfyUI
├── gs_preview_node.py   # Full pipeline implementation
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Limitations

- **Two images only** — the scene is reconstructed as a sparse point cloud; dense 3DGS training is out of scope.
- **No optimization loop** — Gaussians are not refined; colors and scales are fixed at initialization.
- **Unknown intrinsics** — focal length is estimated as `max(H, W) × 1.2`; distorted or fisheye lenses will degrade accuracy.
- **Baseline sensitivity** — very small or very large viewpoint differences between the two images reduce reconstruction quality.

---

## License

MIT
