"""
ComfyUI DCT Custom Nodes
========================
Node 1 · DCT2D          : IMAGE → DCT_SPECTRUM + spectrum preview
Node 2 · DCTMask        : DCT_SPECTRUM + paint mask → masked DCT_SPECTRUM
Node 3 · IDCT2D         : DCT_SPECTRUM → reconstructed IMAGE
"""

import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image

try:
    from scipy.fft import dctn, idctn
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _require_scipy():
    if not _HAS_SCIPY:
        raise RuntimeError(
            "scipy is required for DCT nodes.  "
            "Install with:  pip install scipy"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 · 2D DCT Transform
# ─────────────────────────────────────────────────────────────────────────────
class DCT2DNode:
    """
    Applies a 2-D DCT-II (orthonormal) to each spatial channel of the image.

    Outputs
    -------
    dct_spectrum      : DCT_SPECTRUM tensor [B, H, W, C]  — raw float coefficients
    spectrum_preview  : IMAGE [B, H, W, C] in [0, 1]     — log-scale visualization
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES  = ("DCT_SPECTRUM", "IMAGE")
    RETURN_NAMES  = ("dct_spectrum", "spectrum_preview")
    FUNCTION      = "dct_transform"
    CATEGORY      = "DCT"

    def dct_transform(self, image: torch.Tensor):
        _require_scipy()

        batch_dct     = []
        batch_preview = []

        for b in range(image.shape[0]):
            img = image[b].cpu().numpy().astype(np.float64)   # [H, W, C]

            # 2-D DCT-II along spatial axes, orthonormal
            coeffs = dctn(img, axes=(0, 1), norm="ortho")      # [H, W, C] float64

            batch_dct.append(coeffs.astype(np.float32))

            # Log-magnitude preview with percentile normalisation.
            # Min-max would make the whole image nearly black because the DC
            # component (0,0) dominates by several orders of magnitude.
            # Clipping at the 99th percentile keeps all other coefficients
            # visible and maps the DC spike to pure white.
            log_mag = np.log1p(np.abs(coeffs)).astype(np.float32)
            preview = np.zeros_like(log_mag)
            for c in range(log_mag.shape[2]):
                ch = log_mag[:, :, c]
                lo = float(np.percentile(ch, 0))
                hi = float(np.percentile(ch, 99))
                if hi > lo:
                    preview[:, :, c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)

            batch_preview.append(preview)

        dct_t     = torch.from_numpy(np.stack(batch_dct))
        preview_t = torch.from_numpy(np.stack(batch_preview))

        return (dct_t, preview_t)


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 · DCT Spectrum Mask Editor
# ─────────────────────────────────────────────────────────────────────────────
class DCTMaskNode:
    """
    Applies a user-painted mask to the DCT coefficient tensor.

    The mask is drawn in the node's built-in paint canvas (web/js/dct_nodes.js):
      • Left-click / drag  → paint black  → zero out those frequency components
      • Right-click / drag → paint white  → restore those components
      • Scroll wheel       → brush size
      • Double-click       → clear mask (keep all)

    The DCT2D node's *spectrum_preview* output is automatically used as the
    canvas background once the upstream node has been executed.

    Outputs
    -------
    masked_spectrum : DCT_SPECTRUM — coefficients with painted regions zeroed
    mask_preview    : IMAGE        — grayscale visualisation of the applied mask
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dct_spectrum": ("DCT_SPECTRUM",),
            },
            "optional": {
                # This STRING widget is replaced by the paint canvas in JS.
                # The value is a base64-encoded PNG of the painted mask.
                "mask_data": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES  = ("DCT_SPECTRUM", "IMAGE")
    RETURN_NAMES  = ("masked_spectrum", "mask_preview")
    FUNCTION      = "apply_mask"
    CATEGORY      = "DCT"

    def apply_mask(
        self,
        dct_spectrum: torch.Tensor,
        mask_data: str = "",
    ):
        B, H, W, C = dct_spectrum.shape

        # ── Decode mask ──────────────────────────────────────────────────────
        if not mask_data:
            mask_np = np.ones((H, W), dtype=np.float32)   # keep everything
        else:
            try:
                # Strip data-URL prefix if present
                b64 = mask_data.split(",")[-1]
                png_bytes = base64.b64decode(b64)
                mask_img  = Image.open(BytesIO(png_bytes)).convert("L")
                mask_np   = np.array(mask_img, dtype=np.float32) / 255.0

                # Resize to match spectrum dimensions if necessary
                if mask_np.shape[0] != H or mask_np.shape[1] != W:
                    pil = Image.fromarray(
                        (mask_np * 255).astype(np.uint8), "L"
                    ).resize((W, H), Image.LANCZOS)
                    mask_np = np.array(pil, dtype=np.float32) / 255.0

            except Exception as exc:
                print(f"[DCTMask] mask_data decode error: {exc}")
                mask_np = np.ones((H, W), dtype=np.float32)

        # mask_np: [H, W]  — 0 = black (remove), 1 = white (keep)
        mask_t  = torch.from_numpy(mask_np).to(dct_spectrum.device)
        mask_4d = mask_t.unsqueeze(0).unsqueeze(-1)        # [1, H, W, 1]

        masked = dct_spectrum * mask_4d                    # broadcast over B, C

        # Grayscale preview [1, H, W, 3]
        preview_np = np.stack([mask_np] * 3, axis=-1)[np.newaxis]
        preview_t  = torch.from_numpy(preview_np).to(dct_spectrum.device)

        return (masked, preview_t)


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 · Inverse 2D DCT
# ─────────────────────────────────────────────────────────────────────────────
class IDCT2DNode:
    """
    Applies the inverse 2-D DCT-II (= DCT-III, orthonormal) to reconstruct
    the spatial-domain image from DCT coefficients.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dct_spectrum": ("DCT_SPECTRUM",),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("reconstructed_image",)
    FUNCTION      = "idct_transform"
    CATEGORY      = "DCT"

    def idct_transform(self, dct_spectrum: torch.Tensor):
        _require_scipy()

        batch_imgs = []

        for b in range(dct_spectrum.shape[0]):
            coeffs = dct_spectrum[b].cpu().numpy().astype(np.float64)
            img    = idctn(coeffs, axes=(0, 1), norm="ortho")
            img    = np.clip(img, 0.0, 1.0).astype(np.float32)
            batch_imgs.append(img)

        return (torch.from_numpy(np.stack(batch_imgs)),)
