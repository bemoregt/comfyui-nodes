import torch
import numpy as np
from PIL import Image


def _spectral_residual(gray: np.ndarray) -> np.ndarray:
    """
    Spectral Residual Saliency (Hou & Zhang, CVPR 2007)

    1. 입력 그레이스케일 이미지에 2D FFT 적용
    2. 로그 진폭 스펙트럼 계산: A(f) = log(|F(f)|)
    3. 평균 로그 스펙트럼 추정: Ā(f) = h_n * A(f)  (균일 필터 convolution)
    4. 스펙트럼 잔차:           R(f) = A(f) - Ā(f)
    5. 역 FFT로 현저성 맵 생성: S(x) = g * IFFT(exp(R(f) + iP(f)))²
    """
    # 1. FFT
    f = np.fft.fft2(gray.astype(np.float64))
    f_shift = np.fft.fftshift(f)

    # 2. 로그 진폭 스펙트럼
    amp = np.abs(f_shift)
    log_amp = np.log(amp + 1e-8)

    # 3. 평균 로그 스펙트럼 (균일 필터, 논문 기본값 n=3 → 3×3)
    from scipy.ndimage import uniform_filter
    log_amp_avg = uniform_filter(log_amp, size=3)

    # 4. 스펙트럼 잔차
    residual = log_amp - log_amp_avg

    # 5. 역 FFT (위상 보존)
    phase = np.angle(f_shift)
    reconstructed = np.fft.ifft2(
        np.fft.ifftshift(np.exp(residual + 1j * phase))
    )
    saliency = np.abs(reconstructed) ** 2

    # 6. 가우시안 스무딩 (논문: σ≈8)
    from scipy.ndimage import gaussian_filter
    saliency = gaussian_filter(saliency, sigma=8)

    return saliency


def _apply_colormap(saliency: np.ndarray, colormap: str) -> np.ndarray:
    """현저성 맵을 [0,1]로 정규화 후 컬러맵 적용 → RGB uint8"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = saliency - saliency.min()
    denom = saliency.max() - saliency.min()
    if denom > 0:
        norm = norm / denom
    else:
        norm = np.zeros_like(saliency)

    cmap = plt.get_cmap(colormap)
    colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    return colored


COLORMAPS = ["hot", "inferno", "magma", "jet", "viridis", "plasma", "gray"]


class ImageToSpectralResidualSaliency:
    """
    ComfyUI 커스텀 노드: 이미지 → Spectral Residual Saliency Map

    Hou, X., & Zhang, L. (2007).
    "Saliency Detection: A Spectral Residual Approach."
    CVPR 2007.

    알고리즘:
      1. 입력 이미지를 그레이스케일로 변환
      2. 2D FFT → 로그 진폭 스펙트럼
      3. 균일 필터로 평균 스펙트럼 추정 후 잔차 계산
      4. IFFT로 saliency map 복원 → Gaussian smoothing
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "colormap": (COLORMAPS, {"default": "hot"}),
                "smooth_sigma": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 32.0,
                        "step": 0.5,
                        "tooltip": "Gaussian smoothing σ (0 = 스무딩 없음)",
                    },
                ),
                "avg_filter_size": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 15,
                        "step": 2,
                        "tooltip": "평균 스펙트럼 추정 균일 필터 크기 (홀수 권장)",
                    },
                ),
                "output_mode": (
                    ["colormap", "grayscale", "overlay"],
                    {
                        "default": "colormap",
                        "tooltip": (
                            "colormap: 컬러맵 적용 / "
                            "grayscale: 흑백 / "
                            "overlay: 원본 위에 saliency 오버레이"
                        ),
                    },
                ),
                "overlay_alpha": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "overlay 모드에서 saliency 투명도 (0=원본, 1=saliency만)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("saliency_map",)
    FUNCTION = "process"
    CATEGORY = "image/analysis"

    def process(
        self,
        image: torch.Tensor,
        colormap: str,
        smooth_sigma: float,
        avg_filter_size: int,
        output_mode: str,
        overlay_alpha: float,
    ):
        from scipy.ndimage import uniform_filter, gaussian_filter

        # image: [B, H, W, C] float32 [0,1]
        batch_size = image.shape[0]
        results = []

        for b in range(batch_size):
            img_np = (image[b].numpy() * 255).clip(0, 255).astype(np.uint8)  # [H, W, C]

            # 그레이스케일 변환
            if img_np.shape[2] == 1:
                gray = img_np[:, :, 0].astype(np.float64)
            else:
                pil_img = Image.fromarray(img_np, mode="RGB")
                gray = np.array(pil_img.convert("L")).astype(np.float64)

            # ── Spectral Residual ──────────────────────────────────────────
            f = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f)

            amp = np.abs(f_shift)
            log_amp = np.log(amp + 1e-8)

            log_amp_avg = uniform_filter(log_amp, size=avg_filter_size)
            residual = log_amp - log_amp_avg

            phase = np.angle(f_shift)
            reconstructed = np.fft.ifft2(
                np.fft.ifftshift(np.exp(residual + 1j * phase))
            )
            saliency = np.abs(reconstructed) ** 2

            if smooth_sigma > 0:
                saliency = gaussian_filter(saliency, sigma=smooth_sigma)
            # ──────────────────────────────────────────────────────────────

            # 정규화 [0,1]
            s_min, s_max = saliency.min(), saliency.max()
            if s_max > s_min:
                saliency_norm = (saliency - s_min) / (s_max - s_min)
            else:
                saliency_norm = np.zeros_like(saliency)

            # 출력 모드 처리
            if output_mode == "grayscale":
                out_rgb = np.stack(
                    [saliency_norm, saliency_norm, saliency_norm], axis=-1
                ).astype(np.float32)

            elif output_mode == "colormap":
                out_rgb = _apply_colormap(saliency_norm, colormap).astype(np.float32) / 255.0

            else:  # overlay
                colored = _apply_colormap(saliency_norm, colormap).astype(np.float32) / 255.0
                orig_rgb = img_np[:, :, :3].astype(np.float32) / 255.0
                out_rgb = (1 - overlay_alpha) * orig_rgb + overlay_alpha * colored

            out_rgb = np.clip(out_rgb, 0.0, 1.0)
            results.append(torch.from_numpy(out_rgb))  # [H, W, 3]

        output = torch.stack(results, dim=0)  # [B, H, W, 3]
        return (output,)


NODE_CLASS_MAPPINGS = {
    "ImageToSpectralResidualSaliency": ImageToSpectralResidualSaliency,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToSpectralResidualSaliency": "Image to Spectral Residual Saliency",
}
