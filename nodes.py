"""
ComfyUI custom nodes for Eulerian Video Magnification.

Two nodes are provided:

  EVMMagnification   — works with ComfyUI IMAGE batches (frame tensors).
                       Plug directly into VHS LoadVideo → EVM → VHS SaveVideo.

  EVMVideoFile       — standalone file-based node.
                       Input: absolute path to a video file.
                       Output: path to the magnified video written to
                               ComfyUI's output directory.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from .evm_processor import EVMProcessor

try:
    import folder_paths  # available inside ComfyUI runtime
    _OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:          # running outside ComfyUI (unit tests, etc.)
    _OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(_OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Shared parameter definitions (avoids repetition)
# ──────────────────────────────────────────────────────────────────────────────

_ALPHA_DEF    = ("FLOAT", {"default": 50.0,  "min": 1.0,  "max": 500.0, "step": 1.0,  "display": "slider"})
_LOW_FREQ_DEF = ("FLOAT", {"default": 0.5,   "min": 0.01, "max": 60.0,  "step": 0.01})
_HI_FREQ_DEF  = ("FLOAT", {"default": 3.0,   "min": 0.01, "max": 60.0,  "step": 0.01})
_LEVELS_DEF   = ("INT",   {"default": 4,      "min": 1,    "max": 8,     "step": 1})
_ORDER_DEF    = ("INT",   {"default": 1,      "min": 1,    "max": 5,     "step": 1})
_CS_DEF       = (["rgb", "yiq"],)


# ──────────────────────────────────────────────────────────────────────────────
# Node 1 : IMAGE batch → IMAGE batch  (integrates with VHS / any frame node)
# ──────────────────────────────────────────────────────────────────────────────

class EVMMagnification:
    """
    Eulerian Video Magnification — frame tensor in, frame tensor out.

    Connect:
      VHS LoadVideo  →  EVMMagnification  →  VHS SaveVideo (or Preview)

    Parameters
    ----------
    fps
        Frame rate of the source video.  Must match the actual FPS so that
        the temporal bandpass frequencies are correct.
    alpha
        Amplification factor.  Start around 20–50 for motion; go higher
        (100–300) for subtle colour changes (e.g. pulse).
    low_freq / high_freq
        Bandpass window in Hz.  Micro-expressions: 0.5–4 Hz.
        Heart-rate colour: 0.83–1.0 Hz.
    pyramid_levels
        Depth of the Laplacian pyramid.  4 is a good default for 512×512.
    filter_order
        Butterworth filter order.  1 = smooth roll-off; 3–5 = steep but
        may ring on short clips.
    color_space
        rgb  : amplify all three channels equally (good for motion).
        yiq  : separate luma from chroma; amplify chrominance channels
               more strongly (good for pulse / colour).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":         ("IMAGE",),
                "fps":            ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 0.5}),
                "alpha":          _ALPHA_DEF,
                "low_freq":       _LOW_FREQ_DEF,
                "high_freq":      _HI_FREQ_DEF,
                "pyramid_levels": _LEVELS_DEF,
                "filter_order":   _ORDER_DEF,
                "color_space":    _CS_DEF,
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("images",)
    FUNCTION      = "magnify"
    CATEGORY      = "video/EVM"
    OUTPUT_NODE   = False

    def magnify(
        self,
        images: torch.Tensor,
        fps: float,
        alpha: float,
        low_freq: float,
        high_freq: float,
        pyramid_levels: int,
        filter_order: int,
        color_space: str,
    ) -> tuple[torch.Tensor]:
        frames = images.cpu().numpy()        # (T, H, W, C) float32 [0,1]
        result = EVMProcessor().process(
            frames, fps, alpha, low_freq, high_freq,
            pyramid_levels, filter_order, color_space=color_space,
        )
        return (torch.from_numpy(result),)


# ──────────────────────────────────────────────────────────────────────────────
# Node 2 : Video file path → Video file path  (standalone)
# ──────────────────────────────────────────────────────────────────────────────

class EVMVideoFile:
    """
    Eulerian Video Magnification — video file in, video file out.

    Reads any video format supported by OpenCV, runs EVM, and writes an
    MP4 to ComfyUI's output directory.

    Parameters
    ----------
    video_path
        Absolute path to the source video file.
    output_filename
        Stem of the output file (no extension).  The result is saved as
        <output_dir>/<output_filename>.mp4
    (other parameters — same as EVMMagnification above)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path":      ("STRING", {"default": "", "multiline": False}),
                "alpha":           _ALPHA_DEF,
                "low_freq":        _LOW_FREQ_DEF,
                "high_freq":       _HI_FREQ_DEF,
                "pyramid_levels":  _LEVELS_DEF,
                "filter_order":    _ORDER_DEF,
                "color_space":     _CS_DEF,
                "output_filename": ("STRING", {"default": "evm_output", "multiline": False}),
            }
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("output_video_path",)
    FUNCTION      = "process_video_file"
    CATEGORY      = "video/EVM"

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _read_video(path: str) -> tuple[np.ndarray, float, int, int]:
        """Return (frames_float32, fps, width, height)."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(rgb)
        cap.release()

        if not frames:
            raise RuntimeError(f"No frames decoded from: {path}")

        return np.stack(frames), fps, width, height

    @staticmethod
    def _write_video(
        path: str,
        frames: np.ndarray,
        fps: float,
        width: int,
        height: int,
    ) -> None:
        """Write (T, H, W, C) float32 [0,1] frames to an MP4 file."""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        for frame in frames:
            bgr = cv2.cvtColor(
                (frame * 255).clip(0, 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            writer.write(bgr)
        writer.release()

    # ── main entry ────────────────────────────────────────────────────────────

    def process_video_file(
        self,
        video_path: str,
        alpha: float,
        low_freq: float,
        high_freq: float,
        pyramid_levels: int,
        filter_order: int,
        color_space: str,
        output_filename: str,
    ) -> tuple[str]:
        video_path = video_path.strip()
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"[EVM] Reading {video_path} …")
        frames, fps, W, H = self._read_video(video_path)

        result = EVMProcessor().process(
            frames, fps, alpha, low_freq, high_freq,
            pyramid_levels, filter_order, color_space=color_space,
        )

        out_path = os.path.join(_OUTPUT_DIR, f"{output_filename}.mp4")
        print(f"[EVM] Writing {out_path} …")
        self._write_video(out_path, result, fps, W, H)
        print(f"[EVM] Done → {out_path}")
        return (out_path,)
