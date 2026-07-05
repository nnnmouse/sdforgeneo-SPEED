"""Stable Diffusion WebUI Forge extension implementation for SPEED (Spectral Progressive Diffusion).

Based on https://github.com/howardhx/speed
"""
from __future__ import annotations

import sys
import threading
import torch
import numpy as np

import modules.scripts as scripts
import gradio as gr

from modules import sd_samplers, sd_samplers_common

# Dynamic import helper for WebUI scripts folder environment
try:
    from scripts import speed_utils
except ImportError:
    import speed_utils

# Resolve the KDiffusionSampler module depending on WebUI/Forge engine version
try:
    from modules import sd_samplers_kdiffusion as sdk
except ImportError:
    from modules import sd_samplers as sdk


# =============================================================================
# Power-spectrum presets
# =============================================================================
_PRESETS = {
    "flux":   {"A": 203.615097, "beta": 1.915461},
    "wan21":  {"A": 219.484718, "beta": 2.422687},
    "custom": None,
}


# =============================================================================
# Thread-safe State Management
# =============================================================================
class SpeedSamplerState(threading.local):
    def __init__(self):
        self.active = False
        self.base_sampler = "euler"
        self.transform = "dct"
        self.mode = "delta_optimal"
        self.model_preset = "flux"
        self.scales = [0.5, 1.0]
        self.delta = 0.01
        self.manual_sigmas = [0.85]
        self.spectrum_A = 203.615097
        self.spectrum_beta = 1.915461
        self.seed = 0

speed_state = SpeedSamplerState()


# =============================================================================
# Dynamic extra_args spatial scaling
# =============================================================================
def resize_extra_args(extra_args: dict | None, target_h: int, target_w: int) -> dict:
    """Recursively search for and resize 4D/5D PyTorch tensors in extra_args to target size.
    This step is necessary in Stable Diffusion WebUI/Forge to prevent shape mismatch crashes 
    in deep layers of the U-Net/DiT models when scaling latents.
    """
    if not extra_args:
        return {}
    
    new_args = {}
    for k, v in extra_args.items():
        if isinstance(v, torch.Tensor) and v.ndim in (4, 5):
            # Resize using nearest interpolation to maintain robust performance across varied models
            if v.ndim == 4:
                new_v = torch.nn.functional.interpolate(v, size=(target_h, target_w), mode="nearest")
            else:
                B, C, F, H, W = v.shape
                v_4d = v.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
                v_resized = torch.nn.functional.interpolate(v_4d, size=(target_h, target_w), mode="nearest")
                new_v = v_resized.reshape(B, F, C, target_h, target_w).permute(0, 2, 1, 3, 4)
            new_args[k] = new_v
        elif isinstance(v, dict):
            new_args[k] = resize_extra_args(v, target_h, target_w)
        else:
            new_args[k] = v
    return new_args


# =============================================================================
# Dynamic sampler listing
# =============================================================================
def _list_samplers():
    """Dynamically fetch k-diffusion solvers present in the active environment."""
    try:
        import k_diffusion.sampling as kds
        names = [a[len("sample_"):] for a in dir(kds) if a.startswith("sample_")]
        excluded = {"dpm_fast", "dpm_adaptive", "lcm", "speed"}
        return sorted(n for n in names if n not in excluded)
    except Exception:
        # Fallback list of popular samplers
        return ["euler", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_sde", "uni_pc"]


# =============================================================================
# Callback adaptation
# =============================================================================
def _segment_callback(outer_cb, segment_start_idx: int):
    """Re-base callback step indices to the full schedule."""
    if outer_cb is None:
        return None
    def inner(d):
        d = dict(d)
        d["i"] = d.get("i", 0) + segment_start_idx
        outer_cb(d)
    return inner


# =============================================================================
# Main Custom Sampler Runner
# =============================================================================
@torch.no_grad()
def sample_speed(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """Segmented progressive diffusion solver wrapper for Stable Diffusion WebUI/Forge."""
    import k_diffusion.sampling as kds
    
    # Retrieve base sampler function
    sampler_name = getattr(speed_state, "base_sampler", "euler")
    sampler_fn = getattr(kds, f"sample_{sampler_name}", None)
    if sampler_fn is None:
        sampler_fn = kds.sample_euler

    H_full, W_full = x.shape[-2], x.shape[-1]
    scales = getattr(speed_state, "scales", [0.5, 1.0])

    # If SPEED is not active or has invalid configurations, execute using standard base solver
    if not getattr(speed_state, "active", False) or not scales or len(scales) < 2:
        resized_args = resize_extra_args(extra_args, H_full, W_full)
        return sampler_fn(model, x, sigmas, extra_args=resized_args, callback=callback, disable=disable)

    # Determine resolution transition boundaries
    if speed_state.mode == "delta_optimal":
        transitions = speed_utils._resolve_transitions(
            sigmas, scales, speed_state.delta, speed_state.spectrum_A, speed_state.spectrum_beta, H_full, W_full
        )
    elif speed_state.mode == "manual":
        transitions = speed_utils._resolve_manual(sigmas, scales, speed_state.manual_sigmas)
    else:
        transitions = []

    # If no valid transitions resolve, run the base sampler at full resolution
    if not transitions:
        resized_args = resize_extra_args(extra_args, H_full, W_full)
        return sampler_fn(model, x, sigmas, extra_args=resized_args, callback=callback, disable=disable)

    first_scale = scales[0]
    # DCT-truncate the incoming latent down to the starting coarsest scale
    if first_scale < 1.0:
        x = speed_utils._initial_dct_downscale(x, first_scale)

    sigmas = sigmas.clone()
    segment_starts = [0] + [t[0] for t in transitions]

    # Save WebUI's original hijacked torch module from k_diffusion.sampling
    orig_kds_torch = getattr(kds, "torch", torch)
    try:
        # Restore native torch to bypass WebUI's TorchHijack forcing full-resolution noise shapes
        kds.torch = torch

        for seg_i, seg_start in enumerate(segment_starts):
            seg_end = transitions[seg_i][0] if seg_i < len(transitions) else len(sigmas) - 1
            seg_sigmas = sigmas[seg_start:seg_end + 1]

            if len(seg_sigmas) >= 2:
                cb = _segment_callback(callback, seg_start)
                current_h, current_w = x.shape[-2], x.shape[-1]
                # Interpolate conditional inputs in extra_args to match current resolution of x
                resized_args = resize_extra_args(extra_args, current_h, current_w)

                x = sampler_fn(model, x, seg_sigmas, extra_args=resized_args, callback=cb, disable=disable)

            if seg_i >= len(transitions):
                break

            step_idx, s_i, s_next = transitions[seg_i]
            sigma_at_transition = float(sigmas[step_idx])

            # Expand the latent spectrally and update aligned flow-matching timesteps
            x, t_tilde = speed_utils._expand_and_align_torch(
                x, s_i, s_next, sigma_at_transition,
                transform=speed_state.transform, seed=speed_state.seed + (seg_i + 1) * 10000,
                H_full=H_full, W_full=W_full,
            )

            # Patch only the transition sigma, matching the reference inference loop
            sigmas[step_idx] = float(t_tilde)

    finally:
        # Restore WebUI's original hijacked torch module to prevent breaking other samplers
        kds.torch = orig_kds_torch

    return x


# =============================================================================
# Monkey-patching & Registration
# =============================================================================
# Monkey-patch our Custom Sampler function into the k-diffusion sampling module
import k_diffusion.sampling as kds
kds.sample_speed = sample_speed

def register_speed_sampler():
    """Register the SPEED sampler in the WebUI dropdown list on startup."""
    if any(x.name == "SPEED" for x in sd_samplers.all_samplers):
        return

    try:
        speed_samplers_config = [('SPEED', 'sample_speed', ['speed'], {})]
        
        samplers_data_speed = [
            sd_samplers_common.SamplerData(
                label, 
                lambda model, funcname=funcname: sdk.KDiffusionSampler(funcname, model), 
                aliases, 
                options
            )
            for label, funcname, aliases, options in speed_samplers_config
        ]
        
        sd_samplers.all_samplers += samplers_data_speed
        sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
    except Exception as e:
        print(f"[SPEED Sampler] Error registering custom sampler: {e}", file=sys.stderr)

register_speed_sampler()


# =============================================================================
# WebUI Script Class
# =============================================================================
class SpeedSamplerScript(scripts.Script):
    def title(self) -> str:
        return "SPEED Sampler Extension"

    def show(self, is_img2img: bool) -> bool:
        return scripts.AlwaysVisible

    def ui(self, is_img2img: bool) -> list:
        with gr.Accordion("SPEED (Spectral Progressive Diffusion) Settings", open=False):
            with gr.Row():
                base_sampler = gr.Dropdown(
                    label="Base Sampler",
                    choices=_list_samplers(),
                    value="euler",
                    tooltip="Underlying ODE solver used for each segment.",
                )
                transform = gr.Dropdown(
                    label="Transform Basis",
                    choices=["dct", "fft"],
                    value="dct",
                    tooltip="Spectral basis used at each transition.",
                )
                mode = gr.Dropdown(
                    label="Transition Mode",
                    choices=["delta_optimal", "manual"],
                    value="delta_optimal",
                    tooltip="Whether transitions are calculated via optimal frequency thresholds or manual inputs.",
                )
            with gr.Row():
                model_preset = gr.Dropdown(
                    label="Power-Spectrum Preset",
                    choices=list(_PRESETS.keys()),
                    value="flux",
                    tooltip="Preset power-spectrum coefficients.",
                )
                scales = gr.Textbox(
                    label="Resolution Scales",
                    value="0.5, 1.0",
                    tooltip="Comma-separated resolution fractions ending with 1.0. Example: 0.5, 1.0",
                )
                delta = gr.Slider(
                    label="Delta Threshold",
                    minimum=1e-4,
                    maximum=0.5,
                    step=0.001,
                    value=0.01,
                    tooltip="Noise tolerance parameter for delta-optimal calculation.",
                )
            with gr.Row():
                manual_sigmas = gr.Textbox(
                    label="Manual Sigmas",
                    value="0.85",
                    tooltip="Comma-separated manual transition thresholds (used in manual mode).",
                )
                spectrum_a = gr.Number(
                    label="Spectrum A (Custom)",
                    value=203.615097,
                    tooltip="Manual A parameter (used when preset is custom).",
                )
                spectrum_beta = gr.Number(
                    label="Spectrum Beta (Custom)",
                    value=1.915461,
                    tooltip="Manual Beta parameter (used when preset is custom).",
                )
            with gr.Row():
                seed = gr.Number(
                    label="Transition Noise Seed Offset",
                    value=0,
                    precision=0,
                    tooltip="Offset added to the generation seed to produce spectral expansion noise.",
                )

        # Register the components and their corresponding metadata keys.
        # This tells WebUI to auto-fill these fields when parsing PNG Info.
        self.infotext_fields = [
            (base_sampler, "SPEED Base Sampler"),
            (transform, "SPEED Transform"),
            (mode, "SPEED Mode"),
            (scales, "SPEED Scales"),
            (model_preset, "SPEED Model Preset"),
            (delta, "SPEED Delta"),
            (manual_sigmas, "SPEED Manual Sigmas"),
            (spectrum_a, "SPEED Spectrum A"),
            (spectrum_beta, "SPEED Spectrum Beta"),
            (seed, "SPEED Seed Offset"),
        ]

        return [
            base_sampler, transform, mode, model_preset, scales, delta,
            manual_sigmas, spectrum_a, spectrum_beta, seed
        ]

    def process(
        self, p, base_sampler, transform, mode, model_preset, scales, delta,
        manual_sigmas, spectrum_a, spectrum_beta, seed
    ):
        """WebUI pre-generation hook to extract configurations and populate thread-local state."""
        selected_sampler = getattr(p, "sampler_name", None)
        
        # Verify if 'SPEED' is selected as the active sampler in the main dropdown
        if selected_sampler == "SPEED":
            speed_state.active = True
            speed_state.base_sampler = base_sampler
            speed_state.transform = transform
            speed_state.mode = mode
            speed_state.model_preset = model_preset
            
            # Parse and validate resolution scales input
            try:
                parsed_scales = [float(x.strip()) for x in scales.split(",") if x.strip()]
                if not parsed_scales:
                    parsed_scales = [0.5, 1.0]
                if parsed_scales[-1] != 1.0:
                    parsed_scales.append(1.0)
                parsed_scales = sorted(list(set(parsed_scales)))
                speed_state.scales = parsed_scales
            except Exception:
                speed_state.scales = [0.5, 1.0]
                
            speed_state.delta = float(delta)
            
            # Parse and validate manual sigmas
            try:
                parsed_sigmas = [float(x.strip()) for x in manual_sigmas.split(",") if x.strip()]
                speed_state.manual_sigmas = sorted(parsed_sigmas, reverse=True)
            except Exception:
                speed_state.manual_sigmas = [0.85]
                
            # Populate preset or custom spectrum parameters
            preset = _PRESETS.get(model_preset)
            if preset is not None:
                speed_state.spectrum_A = preset["A"]
                speed_state.spectrum_beta = preset["beta"]
            else:
                speed_state.spectrum_A = float(spectrum_a)
                speed_state.spectrum_beta = float(spectrum_beta)

            # Link noise generator seed to the main generation seed to ensure reproducibility
            base_seed = getattr(p, "seed", 0)
            if base_seed == -1:
                base_seed = 0
            speed_state.seed = int(base_seed + seed)

            # Unconditionally save all parameters to ensure consistent restoration from PNG metadata
            p.extra_generation_params["SPEED Base Sampler"] = base_sampler
            p.extra_generation_params["SPEED Transform"] = transform
            p.extra_generation_params["SPEED Mode"] = mode
            p.extra_generation_params["SPEED Scales"] = scales
            p.extra_generation_params["SPEED Model Preset"] = model_preset
            p.extra_generation_params["SPEED Delta"] = delta
            p.extra_generation_params["SPEED Manual Sigmas"] = manual_sigmas
            p.extra_generation_params["SPEED Spectrum A"] = spectrum_a
            p.extra_generation_params["SPEED Spectrum Beta"] = spectrum_beta
            p.extra_generation_params["SPEED Seed Offset"] = seed
        else:
            # Set inactive to prevent interference with regular samplers
            speed_state.active = False