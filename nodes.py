"""
ComfyUI custom node: Phase-Based Frame Interpolation (2x)

Interpolates video frames using Fourier-transform-based phase differences
on a complex steerable pyramid, producing a 2x frame-rate video with
minimal ghosting compared to naive linear interpolation.
"""

import os
import numpy as np
import cv2
import folder_paths

from .phase_interpolation import interpolate_video_2x

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.webm', '.mkv', '.gif', '.m4v', '.flv')


def _list_input_videos():
    input_dir = folder_paths.get_input_directory()
    if not os.path.isdir(input_dir):
        return []
    return sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    )


class PhaseBasedFrameInterpolation:
    """
    2x Frame Interpolation via Phase-Based Fourier Method.

    Accepts a video file from the ComfyUI input directory,
    inserts one interpolated frame between every pair of consecutive frames,
    and writes the result to the ComfyUI output directory.
    """

    @classmethod
    def INPUT_TYPES(cls):
        videos = _list_input_videos()
        return {
            "required": {
                "video": (
                    videos,
                    {"video_upload": True},
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
    OUTPUT_NODE = True
    FUNCTION = "interpolate"
    CATEGORY = "video/interpolation"
    DESCRIPTION = (
        "2x frame interpolation using Fourier-transform phase differences. "
        "Inserts one synthesized frame between every consecutive pair, "
        "doubling the frame rate while preserving sharpness and structure."
    )

    def interpolate(self, video: str, pyramid_levels: int, output_filename: str):
        # Resolve input path from ComfyUI input directory
        input_dir = folder_paths.get_input_directory()
        video_path = folder_paths.get_annotated_filepath(video) \
            if "[" in video else os.path.join(input_dir, video)

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Resolve output path
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        base, ext = os.path.splitext(output_filename.strip() or "phase_interp_output.mp4")
        if not ext:
            ext = ".mp4"
        out_filename = base + ext
        out_path = os.path.join(output_dir, out_filename)

        # Run phase-based interpolation
        interpolate_video_2x(
            input_path=video_path,
            output_path=out_path,
            num_levels=pyramid_levels,
        )

        # Extract first frame as thumbnail preview for the UI
        previews = []
        cap = cv2.VideoCapture(out_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            thumb_name = base + "_thumb.png"
            thumb_path = os.path.join(output_dir, thumb_name)
            cv2.imwrite(thumb_path, frame)
            previews = [{"filename": thumb_name, "subfolder": "", "type": "output"}]

        return {
            "ui": {
                "video": [
                    {
                        "filename": out_filename,
                        "subfolder": "",
                        "type": "output",
                        "format": "video/mp4",
                    }
                ],
                "images": previews,
            },
            "result": (out_path,),
        }


# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "PhaseBasedFrameInterpolation": PhaseBasedFrameInterpolation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhaseBasedFrameInterpolation": "Phase-Based Frame Interpolation (2x)",
}
