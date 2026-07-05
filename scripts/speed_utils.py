"""Spectral expansion and transition scheduling utilities for SPEED.

Based on https://github.com/howardhx/speed
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np
import torch
from scipy.fft import dctn, idctn

def power_spectrum(omega: float, A: float, beta: float) -> float:
    """Radial power-law spectrum ``P(omega) = A * |omega|**(-beta)``.

    Args:
    - omega: Radial spatial frequency.
    - A: Power-law amplitude (fitted per model).
    - beta: Power-law decay exponent (fitted per model).

    Returns:
    - The power-spectrum value ``P(omega)``.
    """
    return A * abs(omega) ** (-beta)


def activation_time(P_omega: float, delta: float) -> float:
    """Return the activation time for one radial frequency ``omega``.
    This matches Eq. 9 in the paper.

    Args:
    - P_omega: Power-spectrum value ``P(omega)`` at the frequency of interest.
    - delta: Noise-dominated tolerance; smaller ``delta`` delays activation.

    Returns:
    - The activation time ``t_omega`` in ``(0, 1)``.
    """
    if delta >= 1.0:
        raise ValueError(f"delta={delta} >= 1, but we assume the error threshold is < 1.")
    return 1.0 / (1.0 + math.sqrt(delta / (P_omega * (1.0 + P_omega - delta))))


def delta_optimal_transitions(
    scales: Sequence[float],
    delta: float,
    A: float,
    beta: float,
    H: int,
    W: int,
) -> List[float]:
    """Return transition times for adjacent scales. This matches Eq. 10 from the paper.

    Args:
    - scales: Strictly increasing scale list ending at 1.0.
    - delta: Noise-dominated tolerance passed to ``activation_time``.
    - A: Power-law amplitude.
    - beta: Power-law decay exponent.
    - H: Full-resolution latent height.
    - W: Full-resolution latent width.

    Returns:
    - List of transition times ``t*_i`` (length ``len(scales) - 1``).
    """
    validate_scales(scales)
    omega_max = min(H, W) / 2.0
    transitions: List[float] = []
    for i in range(len(scales) - 1):
        omega_i = scales[i] * omega_max
        transitions.append(activation_time(power_spectrum(omega_i, A, beta), delta))
    return transitions


def align_timestep(t: float, r: float) -> float:
    """Return the aligned flow-matching time after spectral noise expansion.
    This matches Eq. 6 of the paper.

    Args:
    - t: Flow-matching time at the resolution transition.
    - r: Resolution scale ratio ``s_{i + 1} / s_i`` of the transition.

    Returns:
    - The aligned flow-matching time ``t_tilde``.
    """
    return t * kappa(t, r)


def kappa(t: float, r: float) -> float:
    """Return the state-rescaling factor after spectral noise expansion.
    This matches Eq. 5 of the paper.

    Args:
    - t: Flow-matching time at the resolution transition.
    - r: Scale ratio ``s_{i + 1} / s_i`` of the transition.

    Returns:
    - The state-rescaling factor ``kappa``.
    """
    return r / (1.0 + (r - 1.0) * t)


def _dct_expand_np(
    x_np: np.ndarray, target_hw: Tuple[int, int], t: float, seed: int,
) -> np.ndarray:
    """DCT spectral noise expansion.

    Args:
    - x_np: Source array; trailing two axes are the spatial grid to expand.
    - target_hw: Target ``(height, width)`` of the expanded grid.
    - t: Noise amplitude for the high-frequency coefficients.
    - seed: Seed for the per-call random generator.

    Returns:
    - The expanded array at ``target_hw`` (float32, same leading axes as ``x_np``).
    """
    H_tgt, W_tgt = target_hw
    H_src, W_src = x_np.shape[-2], x_np.shape[-1]
    if H_tgt < H_src or W_tgt < W_src:
        raise ValueError(
            f"DCT expand: cannot expand to target {target_hw} smaller than "
            f"source ({H_src}, {W_src})."
        )
    rng = np.random.default_rng(seed)
    out = np.empty(x_np.shape[:-2] + (H_tgt, W_tgt), dtype=np.float32)
    for idx in np.ndindex(*x_np.shape[:-2]):
        coeffs_src = dctn(x_np[idx], type=2, norm="ortho")
        big = t * rng.standard_normal((H_tgt, W_tgt)).astype(np.float32)
        big[:H_src, :W_src] = coeffs_src
        out[idx] = idctn(big, type=2, norm="ortho").astype(np.float32)
    return out

def _fft_expand_np(
    x_np: np.ndarray, target_hw: Tuple[int, int], t: float, seed: int,
) -> np.ndarray:
    """FFT spectral noise expansion.

    Args:
    - x_np: Source array; trailing two axes are the spatial grid to expand.
    - target_hw: Target ``(height, width)`` of the expanded grid.
    - t: Noise amplitude for the outer (high-frequency) spectrum.
    - seed: Seed for the per-call random generator.

    Returns:
    - The expanded array at ``target_hw`` (float32, same leading axes as ``x_np``).
    """
    H_tgt, W_tgt = target_hw
    H_src, W_src = x_np.shape[-2], x_np.shape[-1]
    if H_tgt < H_src or W_tgt < W_src:
        raise ValueError(
            f"FFT expand: cannot expand to target {target_hw} smaller than "
            f"source ({H_src}, {W_src})."
        )
    rng = np.random.default_rng(seed)
    pad_h, pad_w = (H_tgt - H_src) // 2, (W_tgt - W_src) // 2
    out = np.empty(x_np.shape[:-2] + (H_tgt, W_tgt), dtype=np.float32)
    for idx in np.ndindex(*x_np.shape[:-2]):
        X_src = np.fft.fftshift(np.fft.fft2(x_np[idx], norm="ortho"))
        nr = rng.standard_normal((H_tgt, W_tgt)).astype(np.float32)
        ni = rng.standard_normal((H_tgt, W_tgt)).astype(np.float32)
        X_big = np.fft.fftshift(t * (nr + 1j * ni) / np.sqrt(2.0))
        X_big[pad_h:pad_h + H_src, pad_w:pad_w + W_src] = X_src
        out[idx] = np.fft.ifft2(np.fft.ifftshift(X_big), norm="ortho").real.astype(np.float32)
    return out


def validate_scales(scales: Sequence[float]) -> None:
    """Validate a strictly increasing resolution scale list ending at 1.0."""
    if len(scales) == 0:
        raise ValueError("list of resolution scales is empty; supply at least one value.")
    if any(s <= 0.0 or s > 1.0 for s in scales):
        raise ValueError(f"every scale must be in (0, 1]; got {list(scales)}")
    if abs(scales[-1] - 1.0) > 1e-6:
        raise ValueError(f"last scale must equal 1.0 (full resolution); got {scales[-1]}")
    for a, b in zip(scales[:-1], scales[1:]):
        if not (a < b):
            raise ValueError(f"scales must be strictly increasing; got {list(scales)}")


def _initial_dct_downscale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Downscale ``x`` by DCT truncation."""
    if scale >= 1.0:
        return x

    H_full, W_full = x.shape[-2], x.shape[-1]
    H_lo, W_lo = round(H_full * scale), round(W_full * scale)

    if x.ndim == 5:
        B, C, T_frames, _, _ = x.shape
        x4 = x.permute(0, 2, 1, 3, 4).reshape(B * T_frames, C, H_full, W_full)
    else:
        x4 = x

    x_np = x4.detach().cpu().float().numpy()
    out_np = np.empty(x_np.shape[:-2] + (H_lo, W_lo), dtype=np.float32)
    for idx in np.ndindex(*x_np.shape[:-2]):
        coeffs = dctn(x_np[idx], type=2, norm="ortho")
        out_np[idx] = idctn(coeffs[:H_lo, :W_lo], type=2, norm="ortho").astype(np.float32)
    out4 = torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype)

    if x.ndim == 5:
        return out4.reshape(B, T_frames, C, H_lo, W_lo).permute(0, 2, 1, 3, 4)
    return out4


def _expand_and_align_torch(
    x: torch.Tensor, s_i: float, s_next: float, t: float,
    transform: str, seed: int, H_full: int, W_full: int,
) -> Tuple[torch.Tensor, float]:
    """Expand a 4D image latent or 5D video latent over its spatial axes."""
    if transform not in ("dct", "fft"):
        raise ValueError(f"transform must be dct|fft, got {transform!r}")
    r = s_next / s_i
    H_tgt = round(s_next * H_full)
    W_tgt = round(s_next * W_full)

    if x.ndim == 5:
        B, C, T_frames, h_lo, w_lo = x.shape
        x4 = x.permute(0, 2, 1, 3, 4).reshape(B * T_frames, C, h_lo, w_lo)
    elif x.ndim == 4:
        x4 = x
    else:
        raise ValueError(f"expected 4D or 5D latent, got shape {tuple(x.shape)}")

    x_np = x4.detach().cpu().float().numpy()
    if transform == "dct":
        expanded = _dct_expand_np(x_np, (H_tgt, W_tgt), t, seed)
    else:
        expanded = _fft_expand_np(x_np, (H_tgt, W_tgt), t, seed)

    rescaled = (kappa(t, r) * expanded).astype(np.float32)
    x4_new = torch.from_numpy(rescaled).to(device=x.device, dtype=x.dtype)

    if x.ndim == 5:
        out = x4_new.reshape(B, T_frames, C, H_tgt, W_tgt).permute(0, 2, 1, 3, 4)
    else:
        out = x4_new

    return out, align_timestep(t, r)


def _resolve_transitions(
    sigmas: torch.Tensor, scales: List[float], delta: float, A: float, beta: float,
    H_full: int, W_full: int,
) -> List[Tuple[int, float, float]]:
    """Return ``(step_idx, s_i, s_next)`` transitions from ``scales`` and ``delta``."""
    if len(scales) < 2:
        return []
    t_stars = delta_optimal_transitions(scales, delta, A, beta, H_full, W_full)
    out: List[Tuple[int, float, float]] = []
    n_steps = len(sigmas) - 1
    for i, (s_old, s_new, t_thr) in enumerate(zip(scales[:-1], scales[1:], t_stars)):
        step_idx = next(
            (j for j in range(n_steps) if float(sigmas[j]) <= t_thr),
            n_steps,
        )
        if step_idx >= n_steps:
            break
        out.append((step_idx, s_old, s_new))
    return out


def _resolve_manual(
    sigmas: torch.Tensor, scales: List[float], manual_sigmas: List[float],
) -> List[Tuple[int, float, float]]:
    """Return transitions from user-specified sigma thresholds."""
    if len(scales) < 2:
        return []
    
    # Enforce or adapt to correct size list to prevent index errors
    expected_len = len(scales) - 1
    if len(manual_sigmas) != expected_len:
        adjusted_manual = list(manual_sigmas)
        if len(adjusted_manual) < expected_len:
            last_val = adjusted_manual[-1] if adjusted_manual else 0.85
            adjusted_manual += [last_val] * (expected_len - len(adjusted_manual))
        else:
            adjusted_manual = adjusted_manual[:expected_len]
    else:
        adjusted_manual = manual_sigmas

    out: List[Tuple[int, float, float]] = []
    n_steps = len(sigmas) - 1
    for s_old, s_new, thr in zip(scales[:-1], scales[1:], adjusted_manual):
        step_idx = next(
            (j for j in range(n_steps) if float(sigmas[j]) <= thr),
            n_steps,
        )
        if step_idx >= n_steps:
            break
        out.append((step_idx, s_old, s_new))
    return out