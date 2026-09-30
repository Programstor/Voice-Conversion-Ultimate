"""
bin/config.py - single source of truth for the whole app.

NOTE: this lives in bin/ (not next to app.py) because the project already has a
`config/` package (config/stylesheet.py); a top-level config.py would shadow or be
shadowed by it.

Everything that used to be repeated across app.py / modules.py / pipeline.py /
training.py lives here once:

  * Paths            - one pathlib-based place for every folder
  * SRProfile        - per-sample-rate architecture (upsample rates, kernels, segment,
                       mel bins, hop). Used by training, export AND loading.
  * DSPConfig        - F0 range, filters, EQ, gates (sample-rate aware)
  * RealtimeConfig   - block / context / crossfade times in milliseconds
  * RealtimeGeometry - frame-aligned buffer sizes derived from the above
  * IndexConfig      - retrieval index size caps / k-means settings
  * TrainConfig      - optimiser / AMP / loader settings
  * CONVERSION_PARAMS- slider specs shared by the Inference and Realtime tabs

Optional overrides: put a `config.toml` next to the app with [dsp], [realtime],
[index] and/or [train] tables, e.g.

    [realtime]
    block_ms = 250
    crossfade_ms = 40

    [index]
    max_vectors = 20000
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional
import tomllib

log = logging.getLogger("rvc.config")


class TaskCancelled(Exception):
    """Raised inside a long-running task when it notices its stop_check() became True.
    UniversalWorker (lib/ui/sbutton.py) treats this as a clean stop, not an error."""

# --------------------------------------------------------------------------- #
# Constants shared by every module
# --------------------------------------------------------------------------- #
FRAME_MS = 10                    # one F0 / mel frame = 10 ms (100 fps) at any model sample rate
FEAT_SR = 16000                 # ContentVec / FCPE input rate
FEAT_HOP = FEAT_SR // 100        # 160 samples per frame at 16 kHz
HUBERT_STRIDE = 320              # 20 ms per ContentVec frame
HUBERT_RECEPTIVE = 400           # samples a ContentVec frame needs
FEAT_DIM = 768                   # ContentVec v2 hidden size
ST_SRS = ["0", "8000", "32000", "44100", "48000", "96000", "192000"]


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def hubert_frames(n_samples_16k: int) -> int:
    """Number of ContentVec (HuBERT) frames produced for n samples at 16 kHz."""
    n = n_samples_16k
    for k, s in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
        n = (n - k) // s + 1
    return max(n, 0)


def pad_right_16k(n_frames: int) -> int:
    """Samples to append (16 kHz) to a window of `n_frames` 10 ms frames so ContentVec yields
    at least ceil(n_frames / 2) frames of 20 ms. Without this the model output is 1-2 frames
    short and would have to be time-stretched to fit."""
    have = n_frames * FEAT_HOP
    need = (math.ceil(n_frames / 2) - 1) * HUBERT_STRIDE + HUBERT_RECEPTIVE
    return max(0, need - have)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Paths:
    root: Path

    @classmethod
    def discover(cls) -> "Paths":
        return cls(Path(os.environ.get("VCU_HOME", Path.cwd())).resolve())

    @property
    def logs(self) -> Path:       return self.root / "logs"
    @property
    def pretrained(self) -> Path: return self.root / "assets" / "pretrained"
    @property
    def models(self) -> Path:     return self.root / "user" / "models"
    @property
    def indices(self) -> Path:    return self.root / "user" / "indices"
    @property
    def seperated(self) -> Path:    return self.root / "user" / "seperated"
    @property
    def converted(self) -> Path:  return self.root / "user" / "converted"
    @property
    def training(self) -> Path:   return self.root / "user" / "training"
    @property
    def dataset(self) -> Path:    return self.root / "user" / "dataset"
    @property
    def style(self) -> Path:      return self.root / "config" / "style.qss"

    def training_run(self, name: str) -> Path:
        return self.training / name

    def ensure(self) -> None:
        for p in (self.logs, self.pretrained, self.models, self.indices,
                  self.seperated, self.converted, self.training, self.dataset):
            p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Sample-rate profiles (RVC v2, 768-dim HuBERT, NSF + f0)
# --------------------------------------------------------------------------- #
_SHARED_ARCH = dict(
    inter_channels=192,
    hidden_channels=192,
    filter_channels=768,
    n_heads=2,
    n_layers=6,
    kernel_size=3,
    resblock="1",
    resblock_kernel_sizes=(3, 7, 11),
    resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    upsample_initial_channel=512,
    spk_embed_dim=109,
    gin_channels=256,
)


@dataclass(frozen=True)
class SRProfile:
    sr: int
    upsample_rates: tuple
    upsample_kernels: tuple
    segment_size: int
    n_mels: int
    filter_length: int = 2048

    def __post_init__(self) -> None:
        if math.prod(self.upsample_rates) != self.hop:
            raise ValueError(
                f"{self.sr} Hz: prod(upsample_rates)={math.prod(self.upsample_rates)} "
                f"but hop={self.hop}"
            )

    @property
    def hop(self) -> int:           return self.sr // 100
    @property
    def spec_channels(self) -> int: return self.filter_length // 2 + 1
    @property
    def pretrained_g(self) -> str:  return f"f0G{self.sr // 1000}k.pth"
    @property
    def pretrained_d(self) -> str:  return f"f0D{self.sr // 1000}k.pth"
    @property
    def titan_g(self) -> str: return f"G-f0{self.sr // 1000}k-TITAN-Medium.pth"
    @property
    def titan_d(self) -> str: return f"D-f0{self.sr // 1000}k-TITAN-Medium.pth"

    def model_kwargs(self, *, p_dropout: float = 0.0, is_half: bool = False) -> dict:
        """Keyword arguments for SynthesizerTrnMs768NSFsid"""
        kw = dict(_SHARED_ARCH)
        kw.update(
            spec_channels=self.spec_channels,
            segment_size=self.segment_size // self.hop,
            p_dropout=p_dropout,
            upsample_rates=list(self.upsample_rates),
            upsample_kernel_sizes=list(self.upsample_kernels),
            sr=self.sr,
            is_half=is_half,
        )
        kw["resblock_kernel_sizes"] = list(kw["resblock_kernel_sizes"])
        kw["resblock_dilation_sizes"] = [list(d) for d in kw["resblock_dilation_sizes"]]
        return kw

    def checkpoint_config(self, p_dropout: float = 0.0) -> list:
        """Positional list stored in exported .pth files (matches SynthesizerTrnMs768NSFsid.__init__)."""
        kw = self.model_kwargs(p_dropout=p_dropout)
        return [
            kw["spec_channels"], kw["segment_size"], kw["inter_channels"], kw["hidden_channels"],
            kw["filter_channels"], kw["n_heads"], kw["n_layers"], kw["kernel_size"], kw["p_dropout"],
            kw["resblock"], kw["resblock_kernel_sizes"], kw["resblock_dilation_sizes"],
            kw["upsample_rates"], kw["upsample_initial_channel"], kw["upsample_kernel_sizes"],
            kw["spk_embed_dim"], kw["gin_channels"], kw["sr"],
        ]


# 32 kHz is intentionally absent: it needs a different upsample layout and has not been validated.
SR_PROFILES = {
    40000: SRProfile(40000, (10, 10, 2, 2), (16, 16, 4, 4), 12800, 125),
    48000: SRProfile(48000, (12, 10, 2, 2), (24, 20, 4, 4), 17280, 128),
}


def get_profile(sr: int) -> SRProfile:
    try:
        return SR_PROFILES[int(sr)]
    except KeyError:
        raise ValueError(f"Unsupported model sample rate {sr}; supported: {sorted(SR_PROFILES)}") from None


# --------------------------------------------------------------------------- #
# DSP
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DSPConfig:
    f0_min: float = 50.0
    f0_max: float = 1100.0
    f0_bins: int = 256
    median_window: int = 5
    smooth_window: int = 3
    highpass_hz: float = 48.0
    # Offline noise gate: a 10 ms frame is "active" when its peak exceeds gate_db; a frame is
    # kept when >= gate_density of the surrounding gate_min_ms window is active.
    gate_db: float = -35.0
    gate_min_ms: int = 60
    gate_density: float = 0.7
    # (center Hz, gain dB, Q): de-box, presence, air
    eq_bands: tuple = ((350.0, -2.5, 0.7), (3500.0, 1.5, 1.0), (8000.0, 3.0, 0.707))
    lowpass_hz: float = 14500.0
    rms_smoothing: float = 0.2
    max_rms_gain: float = 3.0
    # Air shelf: gentle high-frequency boost to restore presence the model rolls off.
    # Set air_gain_db = 0 to disable entirely.
    air_hz: float = 12000.0   # shelf centre frequency
    air_gain_db: float = 4.0   # boost in dB (3 dB is subtle; 6 dB is noticeable)
    air_q: float = 0.707       # Q of the peaking section used as a shelf approximation
    # Offline: real audio taken from before each slice as ContentVec warm-up (not output)
    offline_context_ms: int = 100

    def eq_for(self, sr: int) -> tuple:
        """EQ bands that are safely below Nyquist for this sample rate."""
        return tuple(b for b in self.eq_bands if b[0] < 0.45 * sr)

    def lowpass_for(self, sr: int) -> float:
        """Low-pass cutoff clamped below Nyquist for this sample rate."""
        return min(self.lowpass_hz, 0.45 * sr)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InferenceConfig:
    f0_up_key: int = 0         # transpose semitone integer
    vol_scale: float = 0.25    # rms envelope scaling to match original volume
    protect: float = 0.33      # preserve breath and voiceless segments
    idx_rate: float = 0.75     # model index matching at inference


# --------------------------------------------------------------------------- #
# Realtime
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RealtimeConfig:
    block_ms: int = 500        # audio converted per step (main latency knob)
    extra_ms: int = 100        # left context for ContentVec warm-up (not played)
    crossfade_ms: int = 40     # SOLA blend zone (adds directly to latency)
    search_ms: int = 10        # SOLA phase search range
    gate_db: float = -35.0     # block is silent when no 10 ms frame reaches this RMS level
    hangover_ms: int = 150     # keep converting after the signal drops below the gate
    stream_sr: int = 48000    # preferred stream sample rate (must be a multiple of 100)
    device_block_ms: int = 20  # PortAudio callback size; the worker accumulates whole blocks
    prebuffer_ms: int = 30     # silence queued before the first block: margin against inference jitter
    max_backlog_blocks: int = 2  # drop input (and resync) when the worker falls this far behind
    latency: str = "low"       # PortAudio latency hint
    report_every_s: float = 5.0


@dataclass(frozen=True)
class RealtimeGeometry:
    """All buffer sizes in whole 10 ms frames so every domain (mic / model / 16 kHz) is an
    exact integer. This removes the need to time-stretch each block to make lengths match.

    Window layout (model domain, oldest -> newest):

        | extra (warm-up, discarded) | crossfade + search (SOLA overlap zone) | block (new) |
    """
    mic_sr: int
    tgt_sr: int
    block: int
    extra: int
    crossfade: int
    search: int

    @classmethod
    def build(cls, cfg: RealtimeConfig, mic_sr: int, tgt_sr: int,
              block_ms: Optional[int] = None) -> "RealtimeGeometry":
        if mic_sr % 100 or tgt_sr % 100:
            raise ValueError(f"Sample rates must be multiples of 100 Hz (mic={mic_sr}, model={tgt_sr}); "
                             f"open the stream at 44100 or 48000.")
        f = lambda ms: max(1, round(ms / FRAME_MS))
        return cls(mic_sr, tgt_sr, f(block_ms or cfg.block_ms), f(cfg.extra_ms),
                   f(cfg.crossfade_ms), f(cfg.search_ms))

    # frames
    @property
    def context_frames(self) -> int: return self.extra + self.crossfade + self.search
    @property
    def window_frames(self) -> int:  return self.context_frames + self.block

    # samples per 10 ms
    @property
    def mic_zc(self) -> int: return self.mic_sr // 100
    @property
    def tgt_zc(self) -> int: return self.tgt_sr // 100

    # mic domain
    @property
    def block_mic(self) -> int:   return self.block * self.mic_zc
    @property
    def context_mic(self) -> int: return self.context_frames * self.mic_zc
    @property
    def window_mic(self) -> int:  return self.window_frames * self.mic_zc

    # model domain
    @property
    def extra_tgt(self) -> int:     return self.extra * self.tgt_zc
    @property
    def crossfade_tgt(self) -> int: return self.crossfade * self.tgt_zc
    @property
    def search_tgt(self) -> int:    return self.search * self.tgt_zc
    @property
    def block_tgt(self) -> int:     return self.block * self.tgt_zc
    @property
    def window_tgt(self) -> int:    return self.window_frames * self.tgt_zc

    # 16 kHz domain
    @property
    def window_16k(self) -> int:    return self.window_frames * FEAT_HOP
    @property
    def pad_right_16k(self) -> int: return pad_right_16k(self.window_frames)

    @property
    def algorithmic_latency_ms(self) -> int:
        """Block + crossfade + search; excludes inference time and device buffers."""
        return (self.block + self.crossfade + self.search) * FRAME_MS


# --------------------------------------------------------------------------- #
# Retrieval index
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IndexConfig:
    max_vectors: int = 10000        # index built by "Train Index" is reduced (k-means) to this size
    kmeans_train_max: int = 100000  # frames sampled from the dataset to fit the k-means
    kmeans_iters: int = 10
    max_bank: int = 100000           # older/larger indices are randomly subsampled to this at load
    top_k: int = 8                   # neighbours blended per frame
    seed: int = 1234


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-4
    lr_decay: float = 0.999875     # per epoch, as in RVC: lr = lr * decay ** epoch
    betas: tuple = (0.8, 0.99)
    eps: float = 1e-9
    grad_acc: int = 4
    c_mel: float = 45.0
    c_fm: float = 2.0              # RVC weights the feature-matching loss x2
    c_kl: float = 1.0              
    num_workers: int = 4           # subprocesses for batch data i/o
    amp: str = "fp16"              # "bf16" (no grad scaler needed), "fp16" (uses GradScaler) or "off"
    seed: int = 1234               # each epoch re-seeds with seed + epoch, so resumed runs match
    cache_specs: bool = True       # keep STFT magnitudes on disk (float16) instead of recomputing per clip
    val_fraction: float = 0.05     # clips held out to measure a mel error that is not trained on
    val_max: int = 16
    preview_seconds: float = 3.0
    chart_ema: float = 0.6         # smoothing of the chart only; the CSV keeps raw values (0 = off)
    check_vram: bool = True
    check_ram: bool = True
    vram_headroom_mb: int = 512    # VRAM kept free after the dataset GPU cache (for the batch forward+backward pass)
    ram_headroom_mb: int = 4096    # RAM kept free after the dataset CPU cache (if no GPU or not enough VRAM)
    vram_activation_mib_per_sample_frame: float = 0.75
    vram_fallback_model_params: int = 120_000_000
    cpu_threads: int = 4
    cpu_compile: bool = True
    cpu_compile_mode: str = "max-autotune"
    batches: int = 4
    frequency: int = 10
    epochs:int = 50


# --------------------------------------------------------------------------- #
# Task-button UI (progress overlay / pulse / busy message on TaskManager buttons)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UiConfig:
    overlay_color: tuple = (119, 185, 0, 100)   # RGBA tint of the button progress/pulse overlay
    progress_anim_ms: int = 450                 # progress-bar fill/unfill animation
    pulse_period_ms: int = 800                  # one half-cycle of the realtime "live" pulse
    pulse_alpha_range: tuple = (60, 160)        # alpha at the dim / bright ends of the pulse
    busy_message_ms: int = 3000                 # how long "Busy with X" shows on a blocked button


# --------------------------------------------------------------------------- #
# Aggregate + optional TOML overrides
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AppConfig:
    paths: Paths
    dsp: DSPConfig
    inference: InferenceConfig
    realtime: RealtimeConfig
    index: IndexConfig
    train: TrainConfig
    ui: UiConfig


def _override(obj: Any, table: dict, section: str) -> Any:
    valid = {f.name for f in fields(obj)}
    clean = {}
    for key, value in table.items():
        if key not in valid:
            log.warning("config.toml [%s]: unknown key %r ignored", section, key)
            continue
        default = getattr(obj, key)
        if isinstance(default, tuple):
            value = tuple(tuple(v) if isinstance(v, list) else v for v in value)
        elif isinstance(default, (int, float, str)) and not isinstance(default, bool):
            value = type(default)(value)
        clean[key] = value
    return replace(obj, **clean)


def load_config(path: Optional[Path] = None) -> AppConfig:
    paths = Paths.discover()
    cfg = AppConfig(paths, DSPConfig(), InferenceConfig(), RealtimeConfig(), IndexConfig(), TrainConfig(), UiConfig())
    path = path or (paths.root / "config.toml")
    if path.is_file():
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        cfg = AppConfig(
            paths,
            _override(cfg.dsp, data.get("dsp", {}), "dsp"),
            _override(cfg.inference, data.get("inference", {}), "inference"),
            _override(cfg.realtime, data.get("realtime", {}), "realtime"),
            _override(cfg.index, data.get("index", {}), "index"),
            _override(cfg.train, data.get("train", {}), "train"),
            _override(cfg.ui, data.get("ui", {}), "ui"),
        )
        log.info("Loaded overrides from %s", path)
    return cfg