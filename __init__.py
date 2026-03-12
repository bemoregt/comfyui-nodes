"""
ComfyUI DCT Custom Node Pack
=============================
• DCT2D        – 2-D DCT transform  (IMAGE → DCT_SPECTRUM + preview)
• DCTMask      – frequency mask editor with built-in paint canvas
• IDCT2D       – inverse DCT        (DCT_SPECTRUM → IMAGE)

Install: place this folder inside  ComfyUI/custom_nodes/
Deps:    pip install scipy
"""

from .nodes import DCT2DNode, DCTMaskNode, IDCT2DNode

NODE_CLASS_MAPPINGS = {
    "DCT2D":    DCT2DNode,
    "DCTMask":  DCTMaskNode,
    "IDCT2D":   IDCT2DNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DCT2D":    "DCT 2D Transform",
    "DCTMask":  "DCT Spectrum Mask Editor",
    "IDCT2D":   "Inverse DCT 2D",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
