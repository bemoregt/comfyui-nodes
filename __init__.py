"""
ComfyUI — Eulerian Video Magnification nodes.

Registers two nodes under the "video/EVM" category:

  • EVM Magnification (Frames)     — IMAGE batch → IMAGE batch
  • EVM Magnification (Video File) — file path   → file path
"""

from .nodes import EVMMagnification, EVMVideoFile

NODE_CLASS_MAPPINGS = {
    "EVMMagnification": EVMMagnification,
    "EVMVideoFile":     EVMVideoFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EVMMagnification": "EVM Magnification (Frames)",
    "EVMVideoFile":     "EVM Magnification (Video File)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
