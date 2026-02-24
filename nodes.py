"""
ComfyUI custom node: Phase-Based Frame Interpolation (2x)

Interpolates video frames using Fourier-transform-based phase differences
on a complex steerable pyramid, producing a 2x frame-rate video with
minimal ghosting compared to naive linear interpolation.
"""

import os
import tempfile
import folder_paths

from .phase_interpolation import interpolate_video_2x


class PhaseBasedFrameInterpolation:
    """
    2x Frame Interpolation via Phase-Based Fourier Method.

    Accepts a video file path, inserts one interpolated frame between
    every pair of consecutive frames, and writes the result to an output
    video file.

    Algorithm
    ---------
    1. Convert each frame pair to Lab color space.
    2. Build complex analytic pyramids from the luminance channel.
    3. Compute hierarchically-unwrapped phase differences.
    4. Reconstruct interpolated luminance from amplitude + phase.
    5. Blend chrominance linearly (human vision is less sensitive there).
    6. Assemble output video at 2× the original frame rate.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Absolute path to the input video file (mp4, avi, mov, ...)",
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
                "output_filename": (
                    "STRING",
                    {
                        "default": "phase_interp_output.mp4",
                        "multiline": False,
                        "tooltip": "Output filename (saved in ComfyUI output directory)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)
    FUNCTION = "interpolate"
    CATEGORY = "video/interpolation"
    DESCRIPTION = (
        "2x frame interpolation using Fourier-transform phase differences. "
        "Inserts one synthesized frame between every consecutive pair, "
        "doubling the frame rate while preserving sharpness and structure."
    )

    def interpolate(
        self,
        video_path: str,
        pyramid_levels: int,
        output_filename: str,
    ) -> tuple:
        video_path = video_path.strip()
        if not video_path:
            raise ValueError("video_path must not be empty")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Resolve output directory
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        # Sanitize filename
        if not output_filename.strip():
            output_filename = "phase_interp_output.mp4"
        base, ext = os.path.splitext(output_filename)
        if not ext:
            ext = ".mp4"
        out_path = os.path.join(output_dir, base + ext)

        # Run interpolation
        result = interpolate_video_2x(
            input_path=video_path,
            output_path=out_path,
            num_levels=pyramid_levels,
        )

        return (result,)


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "PhaseBasedFrameInterpolation": PhaseBasedFrameInterpolation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhaseBasedFrameInterpolation": "Phase-Based Frame Interpolation (2x)",
}
