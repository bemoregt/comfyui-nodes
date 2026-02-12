"""
ComfyUI custom node: Fractional Fourier Transform (FrFT) Spectrum
"""

import numpy as np
import torch

from .frft import frft2d_spectrum


class FrFTSpectrumNode:
    """
    Computes the 2D Fractional Fourier Transform of an input image
    and returns the magnitude spectrum as a displayable image.

    Order conventions
    -----------------
    order = 0  : identity (output ≈ input magnitude)
    order = 1  : standard 2D DFT spectrum (like FFT shift)
    order = 2  : spatially-reversed image
    order = 0.5: halfway between spatial and frequency domain
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "order_x": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "display": "slider",
                    },
                ),
                "order_y": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "display": "slider",
                    },
                ),
                "log_scale": ("BOOLEAN", {"default": True}),
                "channel_mode": (
                    ["luminance", "per_channel"],
                    {"default": "luminance"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("spectrum",)
    FUNCTION = "compute"
    CATEGORY = "image/transform"
    DESCRIPTION = (
        "Computes the 2D Fractional Fourier Transform magnitude spectrum. "
        "order=1 produces the standard FFT spectrum; "
        "order=0.5 is halfway between spatial and frequency domains."
    )

    def compute(
        self,
        image: torch.Tensor,
        order_x: float,
        order_y: float,
        log_scale: bool,
        channel_mode: str,
    ) -> tuple:
        """
        Parameters
        ----------
        image        : [B, H, W, C] float32 tensor, values in [0, 1]
        order_x      : FrFT order along x-axis (columns)
        order_y      : FrFT order along y-axis (rows)
        log_scale    : apply log(1 + |F|) before normalizing
        channel_mode : 'luminance' converts to grayscale first,
                       'per_channel' processes each RGB channel independently

        Returns
        -------
        Tuple of one IMAGE tensor [B, H, W, 3] float32 in [0, 1]
        """
        B, H, W, C = image.shape
        img_np = image.cpu().numpy().astype(np.float32)  # [B, H, W, C]

        results = []
        for b in range(B):
            frame = img_np[b]  # (H, W, C)

            if channel_mode == "luminance":
                # Weighted luminance (BT.601)
                if C >= 3:
                    gray = (
                        0.299 * frame[:, :, 0]
                        + 0.587 * frame[:, :, 1]
                        + 0.114 * frame[:, :, 2]
                    )
                else:
                    gray = frame[:, :, 0]

                spec = frft2d_spectrum(gray, order_x, order_y, log_scale)
                # Convert grayscale spectrum to RGB
                out = np.stack([spec, spec, spec], axis=-1)  # (H, W, 3)

            else:  # per_channel
                channels = []
                for c in range(min(C, 3)):
                    ch_spec = frft2d_spectrum(
                        frame[:, :, c], order_x, order_y, log_scale
                    )
                    channels.append(ch_spec)
                # Pad to 3 channels if needed
                while len(channels) < 3:
                    channels.append(channels[-1])
                out = np.stack(channels[:3], axis=-1)  # (H, W, 3)

            results.append(out)

        out_np = np.stack(results, axis=0).astype(np.float32)  # (B, H, W, 3)
        out_tensor = torch.from_numpy(out_np)
        return (out_tensor,)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "FrFTSpectrum": FrFTSpectrumNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FrFTSpectrum": "FrFT Spectrum",
}
