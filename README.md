# SD WebUI Forge SPEED Sampler

Stable Diffusion WebUI Forge Neo extension implementing **SPEED** (Spectral Progressive Diffusion) for faster sampling. It progressively expands the latent resolution during the denoising process, reducing calculation overhead in early steps while preserving visual quality.

This extension is designed for the **[Stable Diffusion WebUI Forge (Neo)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** branch maintained by haoming02.

---

## References & Credits

This WebUI Forge implementation is adapted from and based on the following works:

1. **ComfyUI Implementation:** https://github.com/ruwwww/ComfyUI-SPEED (from which this extension was ported)
2. **Official SPEED Codebase:** https://github.com/howardhx/speed
3. **Project Resources:**
   - [Project Page](https://howardxiao.ca/speed/)
   - [Paper (PDF)](https://howardxiao.ca/speed/paper/paper.pdf)
   - [arXiv Link](https://arxiv.org/abs/2605.18736)

Also many thanks to Google for Gemini Flash 3.5 vibing this specific adaptation to sdneo.

---

## Installation

1. Open your Stable Diffusion WebUI Forge interface.
2. Navigate to the **Extensions** tab, then select **Install from URL**.
3. Paste the following URL (make sure there is **no trailing slash** at the end of the URL, or use the `.git` link):
   ```text
   https://github.com/nnnmouse/sdforgeneo-SPEED.git
4. Restart the WebUI to load the extension.

---

## Usage

1. Select **SPEED** from the main **Sampling method** dropdown in txt2img or img2img.
2. Expand the collapsible accordion labeled **SPEED (Spectral Progressive Diffusion) Settings** at the bottom of the options panel to configure your settings.
3. Generate your image.

---

## Configuration Settings

| Parameter | Type | Default | Description |
|---|---|---|---|
| **Base Sampler** | dropdown | `euler` | The underlying solver used for each segmented generation block (e.g., `euler`, `heun`, `dpmpp_2m`). |
| **Transform Basis** | dropdown | `dct` | The spectral basis used during resolution transition expansions. Options are `dct` and `fft`. |
| **Transition Mode** | dropdown | `delta_optimal` | Choose `delta_optimal` to automatically compute transition thresholds based on VAE power-spectrum characteristics, or `manual` for custom step configurations. |
| **Power-Spectrum Preset** | dropdown | `flux` | Preset parameters matching analyzed model spectra (e.g., `flux`, `wan21`). Set to `custom` to use manual inputs. |
| **Resolution Scales** | text | `0.5, 1.0` | Comma-separated resolution fractions ending at `1.0` (e.g., `0.25, 0.5, 1.0`). |
| **Delta Threshold** | slider | `0.01` | Noise tolerance variable used in `delta_optimal` calculations (Eq. 9 of the paper). Smaller values delay resolution transitions. |
| **Manual Sigmas** | text | `0.85` | Comma-separated sigma thresholds representing transition checkpoints. Only utilized under `manual` transition mode. |
| **Spectrum A / Beta** | float | (various) | Custom power-law coefficients used only when **Power-Spectrum Preset** is set to `custom`. |
| **Transition Noise Seed Offset** | integer | `0` | Offset added to the generation seed to generate noise patterns for spectral-noise padding. |

### Delta-Optimal Mode (Recommended)
Set **Transition Mode** to `delta_optimal`, choose a preset (such as `flux`), and set your preferred **Resolution Scales**. The transition timings are automatically derived using the power spectrum calculations described in the paper.

### Manual Mode
Set **Transition Mode** to `manual` and specify the transition thresholds inside **Manual Sigmas**. Since sigma values decrease as denoising progresses, your sequence should be decreasing (e.g., `0.95, 0.85` for a three-scale setup).

---

## BibTeX

```bibtex
@article{xiao2026spectral,
  author    = {Xiao, Howard and Chao, Brian and Yariv, Lior and Wetzstein, Gordon},
  title     = {Spectral Progressive Diffusion for Efficient Image and Video Generation},
  year      = {2026},
}
```
