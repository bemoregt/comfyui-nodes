try:
    from .gs_preview_node import GaussianSplattingPreviewNode

    NODE_CLASS_MAPPINGS = {
        "GaussianSplattingPreview": GaussianSplattingPreviewNode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "GaussianSplattingPreview": "3DGS Preview (2 Images)",
    }

    print("[ComfyUI-3DGS] Loaded successfully.")

except Exception as e:
    print(f"[ComfyUI-3DGS] Failed to load: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
