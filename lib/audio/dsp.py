"""
lib/audio/dsp.py - all DSP building blocks: filter design, stateful filter chains,
                   noise gate, loudness matching, mel spectrograms, and stereo diffusion.

  * highpass_sos / peaking_sos / lowpass_sos / output_sos  biquad design (RBJ cookbook)
  * StatefulSOS    causal biquad filter with persistent state (no edge transients)
  * OutputChain    unified post-processing chain used by both offline and realtime paths:
                     input high-pass -> model output EQ -> loudness matching -> soft clip
  * noise_gate     10 ms-frame gate with a density criterion (torch, any device)
  * LoudnessMatcher smoothed RMS envelope matching (numpy, no GPU sync)
  * zero_phase_highpass  offline forward-backward high-pass for dataset preprocessing
  * mel_filterbank / compute_spec  Slaney-mel filterbank and STFT spectrogram used by
                     both RVCDataset (training) and TrainStep (loss computation)
  * stereo_diffusion   room-diffusion stereo widener for the offline output path
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from bin.config import DSPConfig, FRAME_MS, SRProfile, db_to_lin


# --------------------------------------------------------------------------- #
# Filter design
# --------------------------------------------------------------------------- #
def highpass_sos(sr: int, hz: float, order: int = 2) -> np.ndarray:
    return signal.butter(order, hz, btype="high", fs=sr, output="sos")


def peaking_sos(sr: int, f0: float, gain_db: float, q: float) -> np.ndarray:
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    b = np.array([1 + alpha * A, -2 * cw, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * cw, 1 - alpha / A])
    return np.concatenate([b / a[0], a / a[0]])[None, :]


def lowpass_sos(sr: int, fc: float, q: float = 0.707) -> np.ndarray:
    w0 = 2.0 * math.pi * fc / sr
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    b = np.array([(1 - cw) / 2, 1 - cw, (1 - cw) / 2])
    a = np.array([1 + alpha, -2 * cw, 1 - alpha])
    return np.concatenate([b / a[0], a / a[0]])[None, :]


def output_sos(dsp: DSPConfig, sr: int) -> np.ndarray:
    """High-pass -> EQ bands -> low-pass SOS stack for `sr`."""
    sections = [highpass_sos(sr, dsp.highpass_hz)]
    sections += [peaking_sos(sr, f, g, q) for f, g, q in dsp.eq_for(sr)]
    sections.append(lowpass_sos(sr, dsp.lowpass_for(sr)))
    return np.vstack(sections)


# --------------------------------------------------------------------------- #
# Stateful biquad
# --------------------------------------------------------------------------- #
class StatefulSOS:
    """Causal second-order-section filter whose state persists between calls."""

    def __init__(self, sos: np.ndarray):
        self.sos = np.ascontiguousarray(sos, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.zi = np.zeros((self.sos.shape[0], 2), dtype=np.float64)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = signal.sosfilt(self.sos, x, zi=self.zi)
        return y.astype(np.float32, copy=False)


def zero_phase_highpass(x: np.ndarray, sr: int, hz: float) -> np.ndarray:
    """Offline zero-phase high-pass (forward-backward). Used in dataset preprocessing."""
    sos = highpass_sos(sr, hz)
    if x.shape[-1] < 64:
        return x.astype(np.float32, copy=False)
    return signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Unified output chain
# --------------------------------------------------------------------------- #
class OutputChain:
    """Stateful post-processing applied to every converted audio block, offline or realtime.

    For both paths the steps are identical:
        1. input high-pass  (on the raw mic/source signal, before the model)
        2. output EQ chain  (high-pass -> peaking EQ -> low-pass, on the converted signal)
        3. loudness match   (smooth RMS gain toward the source level)
        4. soft clip        (np.tanh — protects speakers, same ceiling used in realtime)

    Offline callers pass the full-file source waveform as `ref` once; realtime callers
    pass each mic block. The loudness smoother's inter-block state is preserved either way
    because it lives on this object, not in the caller.
    """

    def __init__(self, dsp: DSPConfig, sr: int):
        self.hp  = StatefulSOS(highpass_sos(sr, dsp.highpass_hz))
        self.eq  = StatefulSOS(output_sos(dsp, sr))
        self.loud = LoudnessMatcher(dsp.rms_smoothing, dsp.max_rms_gain)

    def reset(self) -> None:
        self.hp.reset()
        self.eq.reset()
        self.loud.reset()

    def apply_input_hp(self, x: np.ndarray) -> np.ndarray:
        """High-pass the input (mic / source) signal. Call before feeding the model."""
        return self.hp(x)

    def apply_output(self, ref: np.ndarray, out: np.ndarray, vol_scale: float) -> np.ndarray:
        """EQ + loudness-match + soft-clip the model output. `ref` is the input signal
        used as the loudness reference (same block for realtime, full file for offline)."""
        y = self.eq(out)
        y = self.loud(ref, y, vol_scale)
        return np.tanh(y).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Noise gate
# --------------------------------------------------------------------------- #
def noise_gate(x: torch.Tensor, frame: int, thr_db: float, min_ms: int = 60,
               density: float = 0.7) -> torch.Tensor:
    """Mute stretches that are quiet for most of a `min_ms` window.

    x       : 1-D tensor on any device
    frame   : samples per 10 ms frame at x's sample rate
    thr_db  : a frame is active when its peak exceeds this level (dBFS)
    The mask is faded linearly between frame centres so gating never clicks.
    """
    n = x.shape[-1]
    if n == 0:
        return x
    nf = -(-n // frame)
    pad = nf * frame - n
    xp = F.pad(x, (0, pad)) if pad else x

    peak = xp.view(nf, frame).abs().amax(dim=1)
    active = (peak > db_to_lin(thr_db)).to(x.dtype)

    k = max(1, min_ms // FRAME_MS)
    k += 1 - (k % 2)
    dens = F.avg_pool1d(active.view(1, 1, -1), kernel_size=k, stride=1, padding=k // 2,
                        count_include_pad=False)
    mask = (dens >= density).to(x.dtype)
    mask = F.interpolate(mask, size=nf * frame, mode="linear", align_corners=False).view(-1)
    return (xp * mask)[:n]


# --------------------------------------------------------------------------- #
# Loudness
# --------------------------------------------------------------------------- #
class LoudnessMatcher:
    """Scale converted audio toward the source loudness.

    rate=1 follows the source level fully, rate=0 leaves the converted level alone. The gain
    is smoothed between calls and ramped per sample inside a call, so block boundaries do not
    produce steps. Pure numpy: no GPU synchronisation.
    """

    def __init__(self, smoothing: float = 0.2, max_gain: float = 3.0):
        self.smoothing = smoothing
        self.max_gain = max_gain
        self.prev: Optional[float] = None

    def reset(self) -> None:
        self.prev = None

    def __call__(self, ref: np.ndarray, out: np.ndarray, rate: float) -> np.ndarray:
        if out.size == 0:
            return out
        rms1 = float(np.sqrt(np.mean(np.square(ref, dtype=np.float64)))) if ref.size else 0.0
        rms2 = float(np.sqrt(np.mean(np.square(out, dtype=np.float64))))
        if rms2 < 1e-6:
            return out
        gain = min(max((rms1 + 1e-4) / (rms2 + 1e-4), 0.0), self.max_gain)
        target = gain * rate + (1.0 - rate)
        prev = target if self.prev is None else self.prev
        new = prev * (1.0 - self.smoothing) + target * self.smoothing
        self.prev = new
        return (out * np.linspace(prev, new, out.shape[-1], dtype=np.float32)).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Mel spectrogram (used by both RVCDataset caching and TrainStep loss)
# --------------------------------------------------------------------------- #
def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3.0, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(f >= min_log_hz,
                    min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, f / f_sp)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3.0, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def mel_filterbank(sr: int, n_fft: int, n_mels: int,
                   fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """[n_mels, n_fft // 2 + 1] triangular filters on the Slaney mel scale, area-normalised."""
    fmax = fmax or sr / 2.0
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fft_f = np.linspace(0.0, sr / 2.0, 1 + n_fft // 2)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_f)
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper =  ramps[2:]  / fdiff[1:,  None]
    w = np.maximum(0.0, np.minimum(lower, upper))
    w *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return w.astype(np.float32)


def compute_spec(wav: torch.Tensor, profile: SRProfile) -> torch.Tensor:
    """Linear STFT magnitude [spec_channels, frames] with reflect padding of (n_fft - hop) / 2,
    so frame k is centred on sample k * hop + hop / 2 and lines up with features and F0."""
    n_fft, hop = profile.filter_length, profile.hop
    pad = (n_fft - hop) // 2
    x = F.pad(wav.view(1, 1, -1), (pad, pad),
               mode="reflect" if wav.numel() > pad else "constant").view(-1)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=torch.hann_window(n_fft), center=False, return_complex=True)
    return torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-6)


# --------------------------------------------------------------------------- #
# Stereo diffusion
# --------------------------------------------------------------------------- #
def _lp(x: torch.Tensor, sr: int, cutoff_hz: float) -> torch.Tensor:
    w = max(3, int(round(sr / cutoff_hz)))
    if w % 2 == 0:
        w += 1
    if x.numel() < w:
        return x
    y = x.view(1, 1, -1)
    for _ in range(2):
        y = F.avg_pool1d(y, kernel_size=w, stride=1, padding=w // 2, count_include_pad=False)
    return y.reshape(-1)


def _reflections(x, delays_ms, gains, sr):
    out = torch.zeros_like(x)
    n = x.numel()
    for d_ms, g in zip(delays_ms, gains):
        d = max(1, int(round(d_ms * sr / 1000.0)))
        if d < n:
            out[d:] += g * x[:-d]
    return out


@torch.no_grad()
def stereo_diffusion(out: torch.Tensor, sr: int, strength: float = 3.0) -> torch.Tensor:
    sig = out.detach().to("cpu", torch.float32).flatten().contiguous()
    n   = sig.numel()
    st  = max(0.0, min(float(strength), 5.0))

    if n == 0:
        return torch.zeros((2, 0), dtype=torch.float32)

    # ── 1. Dry center ────────────────────────────────────────────────────────
    dry_l = sig.clone()
    dry_r = sig.clone()

    # ── 2. Room source — pre-damped ──────────────────────────────────────────
    lo = _lp(sig, sr, 1400.0)
    room_src = lo + 0.30 * (sig - lo)
    late_src = _lp(room_src, sr, 700.0) * 0.80

    # ── 3. Early reflections — asymmetric L / R ──────────────────────────────
    ER_L_MS = ( 4.7,  9.1, 15.3, 23.0, 33.5, 47.0, 63.0)
    ER_L_G  = (0.18, 0.14, 0.10, 0.07, 0.05, 0.03, 0.02)
    ER_R_MS = ( 8.2, 14.0, 21.7, 30.5, 42.0, 57.0, 74.0)
    ER_R_G  = (0.12, 0.17, 0.11, 0.06, 0.04, 0.03, 0.02)

    # ── 4. Late reflections ──────────────────────────────────────────────────
    LATE_L_MS = (58.0,  76.0,  96.0, 119.0, 145.0)
    LATE_L_G  = (0.028, 0.022, 0.017, 0.012, 0.008)
    LATE_R_MS = (65.0,  85.0, 107.0, 131.0, 159.0)
    LATE_R_G  = (0.022, 0.029, 0.015, 0.017, 0.009)

    ref_l = (_reflections(room_src, ER_L_MS, ER_L_G, sr)
           + _reflections(late_src, LATE_L_MS, LATE_L_G, sr))
    ref_r = (_reflections(room_src, ER_R_MS, ER_R_G, sr)
           + _reflections(late_src, LATE_R_MS, LATE_R_G, sr))

    # ── 5. Tonal wall-colour (reflections only) ──────────────────────────────
    lo_l = _lp(ref_l, sr, 900.0);  hi_l = ref_l - lo_l
    lo_r = _lp(ref_r, sr, 900.0);  hi_r = ref_r - lo_r
    ref_l = lo_l * 1.08 + hi_l * 0.80
    ref_r = lo_r * 0.92 + hi_r * 1.15

    # ── 6. Mix ───────────────────────────────────────────────────────────────
    wet   = 0.62 * st
    left  = dry_l + wet * ref_l
    right = dry_r + wet * ref_r

    # ── 7. M/S width ─────────────────────────────────────────────────────────
    mid  = 0.5 * (left + right)
    side = 0.5 * (left - right)
    side_lo = _lp(side, sr, 250.0)
    side = 0.20 * side_lo + (side - side_lo) * (1.10 * st)
    left  = mid + side
    right = mid - side

    # ── 8. Peak limiter ──────────────────────────────────────────────────────
    result = torch.stack([left, right])
    peak = result.abs().amax()
    if peak > 0.985:
        result = result * (0.985 / peak)
    return result.contiguous()