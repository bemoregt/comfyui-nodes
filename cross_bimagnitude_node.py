"""
Cross-BiMagnitude Image Matching/Stitching Node for ComfyUI

Cross-Bimagnitude는 신호처리/고차통계(HOS)에서 교차 바이스펙트럼(Cross-Bispectrum)의
크기(Magnitude) 성분을 의미합니다.

Cross-Bispectrum: B_xy(ω1, ω2) = E[X(ω1) · X(ω2) · Y*(ω1 + ω2)]
Cross-Bimagnitude: |B_xy(ω1, ω2)|

이를 이미지 정합(registration)에 적용하여 두 이미지 간의 최적 정렬을 찾고
스티칭(stitching)합니다.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# 핵심 알고리즘
# ---------------------------------------------------------------------------

def compute_fft2(channel: np.ndarray) -> np.ndarray:
    """2D FFT with zero-padding to next power of 2 for efficiency."""
    h, w = channel.shape
    h2 = 1 << (h - 1).bit_length()
    w2 = 1 << (w - 1).bit_length()
    return np.fft.fft2(channel, s=(h2, w2))


def cross_bimagnitude_correlation(ref_ch: np.ndarray, tgt_ch: np.ndarray) -> np.ndarray:
    """
    Cross-Bimagnitude 기반 상관 맵 계산.

    Cross-Bispectrum의 정의:
        B_xy(ω1, ω2) = X(ω1) · X(ω2) · Y*(ω1 + ω2)

    계산 비용을 줄이기 위해 대각선 슬라이스(ω1=ω2) 적분 근사를 사용합니다:
        ICB(ω) ≈ |X(ω)|^(2/3) · |Y(ω)|^(1/3)   (bimagnitude weighting)

    위 가중치를 위상 상관(phase correlation)에 적용합니다:
        C(ω) = [X(ω)·Y*(ω) / |X(ω)·Y*(ω)|] · BM_weight(ω)

    IFFT(C)의 피크 위치 → 최적 평행이동(translation) 오프셋
    """
    X = compute_fft2(ref_ch.astype(np.float64))
    Y = compute_fft2(tgt_ch.astype(np.float64))

    mag_X = np.abs(X)
    mag_Y = np.abs(Y)

    # Cross-bimagnitude weighting: 바이스펙트럼은 세 항의 곱 → 1/3 지수 사용
    bimag_weight = (mag_X ** (2.0 / 3.0)) * (mag_Y ** (1.0 / 3.0))

    # 위상 정규화 (phase-only)
    cross = X * np.conj(Y)
    phase = cross / (np.abs(cross) + 1e-12)

    # 가중 위상 상관
    weighted = phase * bimag_weight

    # 역 FFT → 상관 맵
    corr = np.fft.ifft2(weighted).real
    corr = np.fft.fftshift(corr)
    return corr


def estimate_translation(ref: np.ndarray, tgt: np.ndarray) -> tuple[int, int]:
    """
    RGB(또는 단일 채널) 이미지에서 채널별 cross-bimagnitude 상관 맵을 평균 내어
    최적 평행이동(dy, dx)을 추정합니다.
    """
    if ref.ndim == 2:
        ref = ref[:, :, np.newaxis]
        tgt = tgt[:, :, np.newaxis]

    n_ch = ref.shape[2]
    acc = None

    for c in range(n_ch):
        corr = cross_bimagnitude_correlation(ref[:, :, c], tgt[:, :, c])
        if acc is None:
            acc = corr
        else:
            # corr 크기를 acc에 맞춤 (zero-padding 차이)
            if corr.shape != acc.shape:
                corr = corr[: acc.shape[0], : acc.shape[1]]
            acc = acc + corr

    peak = np.unravel_index(np.argmax(acc), acc.shape)
    ch, cw = acc.shape[0] // 2, acc.shape[1] // 2
    dy = peak[0] - ch
    dx = peak[1] - cw
    return int(dy), int(dx)


# ---------------------------------------------------------------------------
# 블렌딩 헬퍼
# ---------------------------------------------------------------------------

def linear_blend(canvas: np.ndarray, weight: np.ndarray,
                 patch: np.ndarray, y0: int, x0: int) -> None:
    """캔버스의 [y0:y0+h, x0:x0+w] 영역에 patch를 weight로 누적합니다."""
    h, w = patch.shape[:2]
    canvas[y0:y0 + h, x0:x0 + w] += patch * weight[:, :, np.newaxis]


def create_feather_mask(h: int, w: int, feather: int) -> np.ndarray:
    """경계 feather_px 픽셀에서 부드럽게 감쇠하는 마스크를 생성합니다."""
    mask = np.ones((h, w), dtype=np.float32)
    for i in range(feather):
        v = (i + 1) / (feather + 1)
        mask[i, :] = np.minimum(mask[i, :], v)
        mask[-(i + 1), :] = np.minimum(mask[-(i + 1), :], v)
        mask[:, i] = np.minimum(mask[:, i], v)
        mask[:, -(i + 1)] = np.minimum(mask[:, -(i + 1)], v)
    return mask


def blend_images(ref: np.ndarray, tgt: np.ndarray,
                 dy: int, dx: int,
                 blend_mode: str,
                 feather_px: int) -> np.ndarray:
    """
    ref 기준으로 tgt를 (dy, dx) 만큼 이동시킨 후 블렌딩합니다.

    blend_mode:
        "overlay"  - 단순 덮어쓰기 (tgt가 ref 위에)
        "linear"   - 오버랩 영역에서 선형(weight=0.5) 블렌딩
        "feather"  - 오버랩 경계에서 부드러운 페더 블렌딩
    """
    rh, rw = ref.shape[:2]
    th, tw = tgt.shape[:2]
    n_ch = ref.shape[2]

    # 캔버스 범위 계산
    y_min = min(0, dy)
    x_min = min(0, dx)
    y_max = max(rh, dy + th)
    x_max = max(rw, dx + tw)

    canvas_h = y_max - y_min
    canvas_w = x_max - x_min

    canvas = np.zeros((canvas_h, canvas_w, n_ch), dtype=np.float32)
    weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    # Reference 배치 좌표
    ry0 = 0 - y_min
    rx0 = 0 - x_min

    # Target 배치 좌표
    ty0 = dy - y_min
    tx0 = dx - x_min

    feather = max(1, feather_px)

    if blend_mode == "overlay":
        canvas[ry0:ry0 + rh, rx0:rx0 + rw] = ref
        canvas[ty0:ty0 + th, tx0:tx0 + tw] = tgt

    elif blend_mode == "linear":
        r_mask = np.ones((rh, rw), dtype=np.float32)
        t_mask = np.ones((th, tw), dtype=np.float32)
        linear_blend(canvas, r_mask, ref, ry0, rx0)
        weight_sum[ry0:ry0 + rh, rx0:rx0 + rw] += r_mask
        linear_blend(canvas, t_mask, tgt, ty0, tx0)
        weight_sum[ty0:ty0 + th, tx0:tx0 + tw] += t_mask
        safe_w = np.maximum(weight_sum, 1e-8)[:, :, np.newaxis]
        canvas = canvas / safe_w

    elif blend_mode == "feather":
        r_mask = create_feather_mask(rh, rw, feather)
        t_mask = create_feather_mask(th, tw, feather)
        linear_blend(canvas, r_mask, ref, ry0, rx0)
        weight_sum[ry0:ry0 + rh, rx0:rx0 + rw] += r_mask
        linear_blend(canvas, t_mask, tgt, ty0, tx0)
        weight_sum[ty0:ty0 + th, tx0:tx0 + tw] += t_mask
        safe_w = np.maximum(weight_sum, 1e-8)[:, :, np.newaxis]
        canvas = canvas / safe_w

    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# ComfyUI 노드 클래스
# ---------------------------------------------------------------------------

class CrossBimagnitudeStitch:
    """
    ComfyUI Custom Node: Cross-BiMagnitude Image Stitching

    고차통계(HOS)의 교차 바이스펙트럼(Cross-Bispectrum) 크기를 이용하여
    두 이미지 사이의 최적 정렬을 찾고 스티칭합니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "blend_mode": (["feather", "linear", "overlay"],),
                "feather_px": ("INT", {
                    "default": 40,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "display": "slider",
                }),
                "manual_dy": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1,
                }),
                "manual_dx": ("INT", {
                    "default": 0,
                    "min": -4096,
                    "max": 4096,
                    "step": 1,
                }),
                "use_auto_align": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("stitched_image", "offset_dy", "offset_dx")
    FUNCTION = "stitch"
    CATEGORY = "image/Cross-BiMagnitude"
    DESCRIPTION = (
        "Cross-Bimagnitude (교차 바이스펙트럼 크기) 기반으로 두 이미지를 정렬·스티칭합니다. "
        "HOS(고차통계)의 위상 정보를 활용하여 표준 위상 상관보다 노이즈에 강인한 정합을 수행합니다."
    )

    def stitch(
        self,
        reference_image: torch.Tensor,
        target_image: torch.Tensor,
        blend_mode: str,
        feather_px: int,
        manual_dy: int,
        manual_dx: int,
        use_auto_align: bool,
    ):
        # ComfyUI IMAGE 포맷: (B, H, W, C), float32, [0,1]
        # 배치의 첫 번째 프레임 처리
        ref_np = reference_image[0].cpu().numpy()  # (H, W, C)
        tgt_np = target_image[0].cpu().numpy()      # (H, W, C)

        if use_auto_align:
            dy, dx = estimate_translation(ref_np, tgt_np)
        else:
            dy, dx = manual_dy, manual_dx

        result = blend_images(ref_np, tgt_np, dy, dx, blend_mode, feather_px)

        # (H, W, C) → (1, H, W, C)
        result_tensor = torch.from_numpy(result).unsqueeze(0)

        return (result_tensor, dy, dx)


# ---------------------------------------------------------------------------
# 선택적 추가 노드: Cross-Bimagnitude 스펙트럼 시각화
# ---------------------------------------------------------------------------

class CrossBimagnitudeSpectrum:
    """
    두 이미지의 Cross-Bimagnitude 스펙트럼(대각선 슬라이스)을 시각화합니다.
    디버깅·교육 목적으로 사용합니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "channel": (["luminance", "red", "green", "blue"],),
                "log_scale": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("bimagnitude_spectrum", "correlation_map")
    FUNCTION = "visualize"
    CATEGORY = "image/Cross-BiMagnitude"
    DESCRIPTION = "Cross-Bimagnitude 스펙트럼과 상관 맵을 시각화합니다."

    def visualize(
        self,
        reference_image: torch.Tensor,
        target_image: torch.Tensor,
        channel: str,
        log_scale: bool,
    ):
        ref_np = reference_image[0].cpu().numpy()
        tgt_np = target_image[0].cpu().numpy()

        # 채널 선택
        ch_map = {"luminance": None, "red": 0, "green": 1, "blue": 2}
        ch_idx = ch_map[channel]

        if ch_idx is None:
            # Luminance: BT.601 가중치
            ref_ch = (0.299 * ref_np[:, :, 0]
                      + 0.587 * ref_np[:, :, 1]
                      + 0.114 * ref_np[:, :, 2])
            tgt_ch = (0.299 * tgt_np[:, :, 0]
                      + 0.587 * tgt_np[:, :, 1]
                      + 0.114 * tgt_np[:, :, 2])
        else:
            ref_ch = ref_np[:, :, ch_idx]
            tgt_ch = tgt_np[:, :, ch_idx]

        # Cross-bimagnitude 스펙트럼
        X = compute_fft2(ref_ch.astype(np.float64))
        Y = compute_fft2(tgt_ch.astype(np.float64))
        bimag = (np.abs(X) ** (2.0 / 3.0)) * (np.abs(Y) ** (1.0 / 3.0))
        bimag_shifted = np.fft.fftshift(bimag)

        # 상관 맵
        corr = cross_bimagnitude_correlation(ref_ch, tgt_ch)

        def to_image(arr):
            a = arr.astype(np.float64)
            if log_scale:
                a = np.log1p(a - a.min())
            a = a - a.min()
            mx = a.max()
            if mx > 0:
                a = a / mx
            # 원본 이미지 크기에 맞게 리사이즈
            h, w = ref_np.shape[:2]
            a_t = torch.from_numpy(a.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            a_t = F.interpolate(a_t, size=(h, w), mode="bilinear", align_corners=False)
            a_np = a_t.squeeze().numpy()
            # 흑백 → RGB
            rgb = np.stack([a_np, a_np, a_np], axis=-1)
            return torch.from_numpy(rgb).unsqueeze(0)

        spec_tensor = to_image(bimag_shifted)
        corr_tensor = to_image(corr)

        return (spec_tensor, corr_tensor)


# ---------------------------------------------------------------------------
# 노드 등록
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "CrossBimagnitudeStitch": CrossBimagnitudeStitch,
    "CrossBimagnitudeSpectrum": CrossBimagnitudeSpectrum,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CrossBimagnitudeStitch": "Cross-BiMagnitude Stitch",
    "CrossBimagnitudeSpectrum": "Cross-BiMagnitude Spectrum",
}
