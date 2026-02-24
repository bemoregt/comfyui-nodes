"""
ComfyUI custom node: Phase-Based Frame Interpolation (2x)

Input  : IMAGE tensor  [B, H, W, C]  float32  [0, 1]   — batch of video frames
Output : IMAGE tensor  [B*2-1, H, W, C]  float32  [0, 1] — 2x interpolated frames

Connects directly to any Load Video / Save Video node that uses the
standard ComfyUI IMAGE batch format.
"""

import numpy as np
import torch

from .phase_interpolation import interpolate_batch_2x


class PhaseBasedFrameInterpolation:
    """
    2x Frame Interpolation via Phase-Based Fourier Method.

    Takes a batch of video frames as a standard ComfyUI IMAGE tensor,
    inserts one phase-interpolated frame between every consecutive pair,
    and returns the doubled frame batch as an IMAGE tensor.

    Connect:  Load Video → [this node] → Save Video / Preview
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip": "Batch of video frames [B, H, W, C] from a Load Video node.",
                    },
                ),
                "pyramid_levels": (
                    "INT",
                    {
                        "default": 6,
                        "min": 3,
                        "max": 10,
                        "step": 1,
                        "tooltip": (
                            "Number of complex pyramid levels. "
                            "More levels = finer phase unwrapping but slower."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "interpolate"
    CATEGORY = "video/interpolation"
    DESCRIPTION = (
        "2x frame interpolation using Fourier-transform phase differences. "
        "Accepts an IMAGE batch from a Load Video node, inserts one synthesized "
        "frame between every consecutive pair, and returns the 2x frame batch. "
        "Connect the output to a Save Video or Preview node."
    )

    def interpolate(self, images: torch.Tensor, pyramid_levels: int):
        # images: [B, H, W, C] float32 in [0, 1]
        B, H, W, C = images.shape
        if B < 2:
            raise ValueError("Need at least 2 frames for interpolation.")

        # Convert tensor to list of uint8 RGB numpy arrays
        np_frames = (images.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        frame_list = [np_frames[i] for i in range(B)]

        print(f"[PhaseInterp] Input: {B} frames  Output: {B * 2 - 1} frames")

        # Run phase-based 2x interpolation
        out_list = interpolate_batch_2x(frame_list, num_levels=pyramid_levels)

        # Convert back to [B_out, H, W, C] float32 tensor in [0, 1]
        out_np = np.stack(out_list, axis=0).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(out_np)

        return (out_tensor,)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "PhaseBasedFrameInterpolation": PhaseBasedFrameInterpolation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhaseBasedFrameInterpolation": "Phase-Based Frame Interpolation (2x)",
}
