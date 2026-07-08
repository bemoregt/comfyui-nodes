import json
from collections import deque

import cv2
import numpy as np
import torch


def _tensor_to_gray(img_tensor):
    """img_tensor: (H, W, C) float32 in [0,1] -> (uint8 RGB image, uint8 gray image)."""
    img = np.ascontiguousarray((img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8))
    if img.shape[-1] == 1:
        gray = img[..., 0]
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img, gray


def _detect_segments(gray):
    """Return (Nx4 float32 array of x1,y1,x2,y2, engine name). Tries classic LSD first,
    falls back to FLD since cv2.createLineSegmentDetector is patent-restricted and
    missing/broken on some OpenCV builds."""
    lines = None
    engine = None
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        lines = lsd.detect(gray)[0]
        engine = "LSD"
    except (AttributeError, cv2.error):
        lines = None

    if lines is None:
        if not hasattr(cv2, "ximgproc"):
            raise RuntimeError(
                "cv2.createLineSegmentDetector is unavailable on this OpenCV build and "
                "cv2.ximgproc is missing for the FLD fallback. Install opencv-contrib-python."
            )
        fld = cv2.ximgproc.createFastLineDetector()
        lines = fld.detect(gray)
        engine = "FLD"

    if lines is None:
        return np.zeros((0, 4), dtype=np.float32), engine or "none"
    return lines.reshape(-1, 4).astype(np.float32), engine


def _chain_segments(segs, min_length=5.0, max_gap=0.0):
    """Greedily chain disjoint line segments end-to-end into a single ordered polyline,
    always extending from whichever open end (head or tail) is closest to an
    unused segment's endpoint. Assumes the image contains one dominant curve."""
    segs = [s for s in segs if np.hypot(s[2] - s[0], s[3] - s[1]) >= min_length]
    if not segs:
        return np.zeros((0, 2), dtype=np.float32)

    segs = sorted(segs, key=lambda s: np.hypot(s[2] - s[0], s[3] - s[1]), reverse=True)
    starts = [np.array([s[0], s[1]], dtype=np.float32) for s in segs]
    ends = [np.array([s[2], s[3]], dtype=np.float32) for s in segs]

    n = len(segs)
    used = [False] * n
    used[0] = True
    chain = deque([starts[0], ends[0]])
    remaining = n - 1

    while remaining > 0:
        head, tail = chain[0], chain[-1]
        best = None  # (dist, idx, end, flip)
        for i in range(n):
            if used[i]:
                continue
            p1, p2 = starts[i], ends[i]
            for dist, end, flip in (
                (np.linalg.norm(p1 - tail), "tail", False),
                (np.linalg.norm(p2 - tail), "tail", True),
                (np.linalg.norm(p1 - head), "head", True),
                (np.linalg.norm(p2 - head), "head", False),
            ):
                if best is None or dist < best[0]:
                    best = (dist, i, end, flip)

        dist, idx, end, flip = best
        if max_gap > 0.0 and dist > max_gap:
            break

        used[idx] = True
        p1, p2 = starts[idx], ends[idx]
        if end == "tail":
            if flip:
                chain.append(p2); chain.append(p1)
            else:
                chain.append(p1); chain.append(p2)
        else:
            if flip:
                chain.appendleft(p1); chain.appendleft(p2)
            else:
                chain.appendleft(p2); chain.appendleft(p1)
        remaining -= 1

    return np.array(chain, dtype=np.float32)


def _resample_polyline(points, num_samples):
    """Resample a polyline to num_samples evenly spaced points by arc length,
    which DCT smoothing requires (it assumes uniform sampling)."""
    if len(points) < 2:
        return np.zeros((0, 2), dtype=np.float32)
    deltas = np.diff(points, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_len = cumlen[-1]
    if total_len <= 1e-6:
        return np.repeat(points[:1], num_samples, axis=0)
    sample_dist = np.linspace(0, total_len, num_samples)
    xs = np.interp(sample_dist, cumlen, points[:, 0])
    ys = np.interp(sample_dist, cumlen, points[:, 1])
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _dct_smooth(coords_1d, num_coeffs):
    """Keep the first num_coeffs DCT-II coefficients (low frequencies) and
    reconstruct via inverse DCT -> smoothed / compressed curve axis."""
    src = coords_1d.reshape(-1, 1).astype(np.float32)
    coeffs = cv2.dct(src)
    kept = coeffs.copy()
    kept[num_coeffs:] = 0
    smoothed = cv2.idct(kept).flatten()
    return smoothed, coeffs.flatten()


def _parse_color(color_str):
    try:
        r, g, b = (int(v.strip()) for v in color_str.split(","))
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    except Exception:
        return (255, 0, 0)


def _draw_curve(base_rgb_uint8, points, mode, color, thickness):
    if mode == "blank_black":
        canvas = np.zeros_like(base_rgb_uint8)
    elif mode == "blank_white":
        canvas = np.full_like(base_rgb_uint8, 255)
    else:
        canvas = base_rgb_uint8.copy()
    if len(points) >= 2:
        pts = points.round().astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return canvas


class LSDDCTCurveNode:
    """Detects open-curve line structure in an image with LSD (falling back to FLD),
    chains the resulting line segments into a single ordered polyline, and smooths /
    compresses it by truncating its DCT coefficients."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_dct_coeffs": ("INT", {"default": 16, "min": 2, "max": 512, "step": 1}),
                "num_samples": ("INT", {"default": 256, "min": 16, "max": 4096, "step": 1}),
                "min_segment_length": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.5}),
                "max_gap": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5000.0, "step": 1.0}),
                "canvas_mode": (["overlay_on_original", "blank_black", "blank_white"],),
                "line_thickness": ("INT", {"default": 2, "min": 1, "max": 20, "step": 1}),
                "line_color": ("STRING", {"default": "255,0,0"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "curve_json")
    FUNCTION = "process"
    CATEGORY = "curve/detection"

    def process(
        self,
        image,
        num_dct_coeffs,
        num_samples,
        min_segment_length,
        max_gap,
        canvas_mode,
        line_thickness,
        line_color,
    ):
        num_dct_coeffs = min(num_dct_coeffs, num_samples)
        color = _parse_color(line_color)

        out_images = []
        out_records = []

        for b in range(image.shape[0]):
            rgb_uint8, gray = _tensor_to_gray(image[b])
            segs, engine = _detect_segments(gray)
            chain_pts = _chain_segments(segs, min_length=min_segment_length, max_gap=max_gap)

            if len(chain_pts) >= 2:
                resampled = _resample_polyline(chain_pts, num_samples)
                smoothed_x, dct_x = _dct_smooth(resampled[:, 0], num_dct_coeffs)
                smoothed_y, dct_y = _dct_smooth(resampled[:, 1], num_dct_coeffs)
                smoothed_pts = np.stack([smoothed_x, smoothed_y], axis=1)
            else:
                smoothed_pts = np.zeros((0, 2), dtype=np.float32)
                dct_x = dct_y = np.zeros((0,), dtype=np.float32)

            canvas = _draw_curve(rgb_uint8, smoothed_pts, canvas_mode, color, line_thickness)
            out_images.append(torch.from_numpy(canvas.astype(np.float32) / 255.0))

            out_records.append({
                "engine": engine,
                "num_raw_segments": int(len(segs)),
                "num_chained_points": int(len(chain_pts)),
                "num_samples": int(num_samples),
                "num_dct_coeffs_kept": int(num_dct_coeffs),
                "smoothed_curve": np.round(smoothed_pts, 3).tolist(),
                "dct_coeffs_x": np.round(dct_x[:num_dct_coeffs], 4).tolist(),
                "dct_coeffs_y": np.round(dct_y[:num_dct_coeffs], 4).tolist(),
            })

        out_batch = torch.stack(out_images, dim=0)
        return (out_batch, json.dumps(out_records))


NODE_CLASS_MAPPINGS = {
    "LSDDCTCurve": LSDDCTCurveNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LSDDCTCurve": "LSD -> DCT Curve Smoother",
}
