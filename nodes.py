import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from io import BytesIO
from PIL import Image


COLORMAPS = ["inferno", "magma", "viridis", "plasma", "hot", "cool", "gray"]


def _to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """waveform: [1, C, T] → [T] mono float tensor"""
    audio = waveform.squeeze(0)  # [C, T]
    if audio.ndim == 2:
        audio = audio.mean(dim=0)  # mono
    return audio.float()


def _compute_spectrogram(
    audio: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    use_mel: bool,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """STFT 또는 Mel-spectrogram 계산 후 dB 스케일 numpy 배열 반환 [freq, time]"""
    try:
        import torchaudio
        import torchaudio.transforms as T

        if use_mel:
            transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                f_min=fmin,
                f_max=fmax if fmax > 0 else sample_rate / 2,
                power=2.0,
            )
            spec = transform(audio)  # [n_mels, T]
        else:
            transform = T.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                power=2.0,
            )
            spec = transform(audio)  # [F, T]

        to_db = T.AmplitudeToDB(stype="power", top_db=80)
        spec_db = to_db(spec).numpy()

    except ImportError:
        # torchaudio 없을 경우 numpy STFT fallback
        audio_np = audio.numpy()
        window = np.hanning(win_length)
        step = hop_length
        frames = []
        for start in range(0, max(len(audio_np) - win_length + 1, 1), step):
            frame = audio_np[start: start + win_length]
            if len(frame) < win_length:
                frame = np.pad(frame, (0, win_length - len(frame)))
            frames.append(np.fft.rfft(frame * window))
        spec = np.abs(np.array(frames)).T ** 2  # [F, T]
        spec_db = 10 * np.log10(np.maximum(spec, 1e-10))
        spec_db = np.clip(spec_db, spec_db.max() - 80, spec_db.max())

    return spec_db


def _render_spectrogram(
    spec_db: np.ndarray,
    sample_rate: int,
    hop_length: int,
    use_mel: bool,
    fmin: float,
    fmax: float,
    colormap: str,
    width: int,
    height: int,
    show_axes: bool,
) -> torch.Tensor:
    """스펙트로그램 배열을 matplotlib으로 렌더링 → [1, H, W, 3] float tensor"""
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    n_frames = spec_db.shape[1]
    duration = n_frames * hop_length / sample_rate
    n_freqs = spec_db.shape[0]
    freq_max = fmax if (use_mel and fmax > 0) else (sample_rate / 2)

    extent = [0, duration, fmin, freq_max]
    im = ax.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=colormap,
    )

    if show_axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mel Frequency (Hz)" if use_mel else "Frequency (Hz)")
        plt.colorbar(im, ax=ax, format="%+2.0f dB", pad=0.02)
        plt.title("Mel Spectrogram" if use_mel else "Spectrogram")
        plt.tight_layout()
    else:
        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB").resize((width, height), Image.LANCZOS)
    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W, 3]


class AudioToSpectrogram:
    """
    ComfyUI 커스텀 노드: 오디오 → 스펙트로그램 이미지
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "use_mel": ("BOOLEAN", {"default": True, "label_on": "Mel", "label_off": "Linear"}),
                "n_fft": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "hop_length": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "win_length": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "n_mels": ("INT", {"default": 128, "min": 32, "max": 512, "step": 16}),
                "fmin": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8000.0, "step": 10.0}),
                "fmax": ("FLOAT", {"default": 8000.0, "min": 0.0, "max": 24000.0, "step": 100.0,
                                   "tooltip": "0 = Nyquist (sample_rate / 2)"}),
                "colormap": (COLORMAPS, {"default": "inferno"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "height": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 64}),
                "show_axes": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("spectrogram",)
    FUNCTION = "process"
    CATEGORY = "audio"

    def process(
        self,
        audio,
        use_mel,
        n_fft,
        hop_length,
        win_length,
        n_mels,
        fmin,
        fmax,
        colormap,
        width,
        height,
        show_axes,
    ):
        waveform = audio["waveform"]   # [1, C, T]
        sample_rate = audio["sample_rate"]

        mono = _to_mono(waveform)

        spec_db = _compute_spectrogram(
            mono, sample_rate, n_fft, hop_length, win_length,
            use_mel, n_mels, fmin, fmax,
        )

        image = _render_spectrogram(
            spec_db, sample_rate, hop_length,
            use_mel, fmin, fmax,
            colormap, width, height, show_axes,
        )

        return (image,)


NODE_CLASS_MAPPINGS = {
    "AudioToSpectrogram": AudioToSpectrogram,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioToSpectrogram": "Audio to Spectrogram",
}
