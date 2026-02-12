# ComfyUI Audio to Spectrogram

A ComfyUI custom node that converts an audio input into a spectrogram image.

## Features

- Supports both **Mel** and **Linear** (STFT) spectrograms
- dB-scaled output (top 80 dB dynamic range)
- 7 built-in colormaps: `inferno`, `magma`, `viridis`, `plasma`, `hot`, `cool`, `gray`
- Configurable frequency range, FFT parameters, and output resolution
- Optional axes, colorbar, and title overlay
- Uses `torchaudio` when available; falls back to NumPy STFT automatically

## Installation

Copy this directory into ComfyUI's `custom_nodes` folder:

```
ComfyUI/
└── custom_nodes/
    └── CustomNode_Bispectrum/
        ├── __init__.py
        ├── nodes.py
        └── pyproject.toml
```

Then restart ComfyUI. The node will appear under the **audio** category.

## Dependencies

| Package | Required | Notes |
|---|---|---|
| `torch` | Yes | Comes with ComfyUI |
| `matplotlib` | Yes | Used for rendering |
| `Pillow` | Yes | Image conversion |
| `numpy` | Yes | Array operations |
| `torchaudio` | Recommended | Faster STFT / Mel transforms |

Install recommended dependencies:

```bash
pip install torchaudio matplotlib Pillow
```

## Node: Audio to Spectrogram

**Category:** `audio`

### Inputs

| Parameter | Type | Default | Description |
|---|---|---|---|
| `audio` | AUDIO | — | ComfyUI audio input |
| `use_mel` | BOOLEAN | `True` | `True` = Mel spectrogram, `False` = Linear spectrogram |
| `n_fft` | INT | `2048` | FFT window size (256–8192) |
| `hop_length` | INT | `512` | Frame hop size in samples (64–2048) |
| `win_length` | INT | `2048` | Analysis window length (256–8192) |
| `n_mels` | INT | `128` | Number of Mel filter banks (Mel mode only) |
| `fmin` | FLOAT | `0.0` | Minimum frequency in Hz |
| `fmax` | FLOAT | `8000.0` | Maximum frequency in Hz (set to `0` for Nyquist) |
| `colormap` | COMBO | `inferno` | Matplotlib colormap |
| `width` | INT | `1024` | Output image width in pixels |
| `height` | INT | `512` | Output image height in pixels |
| `show_axes` | BOOLEAN | `True` | Show time/frequency axes, colorbar, and title |

### Output

| Name | Type | Shape | Description |
|---|---|---|---|
| `spectrogram` | IMAGE | `[1, H, W, 3]` | Float32 RGB image, values in `[0, 1]` |

## Example Workflow

1. Add a **Load Audio** node and load any audio file.
2. Connect its output to the `audio` input of **Audio to Spectrogram**.
3. Connect `spectrogram` to a **Preview Image** or **Save Image** node.
4. Adjust `use_mel`, `fmax`, and `colormap` to taste.

## License

MIT
