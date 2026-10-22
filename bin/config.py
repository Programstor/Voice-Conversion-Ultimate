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
FEAT_SR = 16000                  # ContentVec / FCPE input rate
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
    # Lowest pitch the F0 extractor will detect                                  (50.0 Hz)
    f0_min: float = 50.0
    # Highest pitch the F0 extractor will detect                                 (1100.0 Hz)
    f0_max: float = 1100.0
    # Quantisation steps in the pitch embedding lookup table                     (256)
    f0_bins: int = 256
    # Median filter half-width over F0 frames; removes single-frame spikes       (5)
    median_window: int = 5
    # Gaussian/box smooth half-width applied after median; rounds pitch curves   (3)
    smooth_window: int = 3
    # High-pass cutoff below which mic rumble / DC is removed before features    (48.0 Hz)
    highpass_hz: float = 48.0
    # Offline noise gate: a 10 ms frame is "active" when its peak exceeds gate_db; a frame is
    # kept when >= gate_density of the surrounding gate_min_ms window is active.
    # RMS level a frame must reach to be considered speech, not silence          (-35.0 dB)
    gate_db: float = -35.0
    # Window over which gate_density is measured                                 (60 ms)
    gate_min_ms: int = 60
    # Fraction of frames in that window that must be active to keep the frame    (0.7)
    gate_density: float = 0.7
    # Parametric EQ applied to output: (center Hz, gain dB, Q) — de-box, presence, air
    #   350 Hz / -2.5 dB removes boxy resonance
    #   3500 Hz / +1.5 dB adds presence and articulation
    #   8000 Hz / +3.0 dB adds air and openness
    eq_bands: tuple = ((350.0, -2.5, 0.7), (3500.0, 1.5, 1.0), (8000.0, 3.0, 0.707))
    # Low-pass cutoff; clamped to 0.45 × sr at runtime to stay below Nyquist    (14500.0 Hz)
    lowpass_hz: float = 14500.0
    # EMA coefficient for RMS envelope follower (higher = slower response)       (0.2)
    rms_smoothing: float = 0.2
    # Maximum gain the RMS normaliser can apply before it clips instead of boosts (3.0 ×)
    max_rms_gain: float = 3.0
    # Real audio prepended to each offline slice so ContentVec has warm-up context (100 ms)
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
    # Pitch transpose in semitones; 0 = no shift, +12 = one octave up            (0)
    f0_up_key: int = 0
    # RMS envelope scaling applied to the output to match the source volume       (0.25)
    vol_scale: float = 0.25
    # How much of the original unvoiced signal (breath, fricatives) is kept;
    # 0 = replace everything, 1 = keep everything unvoiced                        (0.33)
    protect: float = 0.33
    # Blend ratio between model output (0.0) and nearest-neighbour index match (1.0) (0.75)
    idx_rate: float = 0.75


# --------------------------------------------------------------------------- #
# Realtime
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RealtimeConfig:
    # Audio converted per inference step; the dominant latency term              (500 ms)
    block_ms: int = 500
    # Left-context prepended to each block for ContentVec warm-up; not played    (100 ms)
    extra_ms: int = 100
    # SOLA blend zone between consecutive blocks; adds directly to latency       (40 ms)
    crossfade_ms: int = 40
    # SOLA phase-alignment search range around the crossfade boundary            (10 ms)
    search_ms: int = 10
    # A whole block is skipped when no 10 ms frame in it exceeds this RMS level  (-35.0 dB)
    gate_db: float = -35.0
    # Conversion continues this long after the signal drops below the gate       (150 ms)
    hangover_ms: int = 150
    # Preferred PortAudio stream rate; must be a multiple of 100                 (48000 Hz)
    stream_sr: int = 48000
    # PortAudio callback size; the realtime worker accumulates whole blocks       (20 ms)
    device_block_ms: int = 20
    # Silence pre-queued before the first block to absorb first-inference jitter  (30 ms)
    prebuffer_ms: int = 30
    # Input blocks queued before the worker drops them and resyncs               (2)
    max_backlog_blocks: int = 2
    # PortAudio suggested latency hint: "low", "high", or a float in seconds     ("low")
    latency: str = "low"
    # How often the realtime worker logs its throughput stats                    (5.0 s)
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
    # Built index is reduced to this many vectors via k-means to cap memory      (10000)
    max_vectors: int = 10000
    # Frames sampled from the dataset to train the k-means quantiser             (100000)
    kmeans_train_max: int = 100000
    # k-means Lloyd iterations; more = tighter clusters, slower build            (10)
    kmeans_iters: int = 10
    # Indices larger than this are randomly subsampled when loaded at inference  (100000)
    max_bank: int = 100000
    # Nearest neighbours blended into each output frame during index retrieval   (8)
    top_k: int = 8
    # Random seed for k-means and subsampling; fix for reproducible index builds (1234)
    seed: int = 1234


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainConfig:
    # Peak learning rate for both G and D Adam optimisers                        (1e-4)
    lr: float = 1e-4
    # Multiplicative LR decay applied once per epoch: lr × decay^epoch          (0.999875)
    lr_decay: float = 0.999875
    # Adam β₁ (gradient EMA) and β₂ (squared-gradient EMA)                     ((0.8, 0.99))
    betas: tuple = (0.8, 0.99)
    # Adam ε; prevents division by zero in the adaptive step                    (1e-9)
    eps: float = 1e-9
    # Gradient accumulation steps; effective batch = batches × grad_acc          (4)
    grad_acc: int = 4
    # Weight of the mel-spectrogram reconstruction loss                          (45.0)
    c_mel: float = 45.0
    # Weight of the feature-matching (discriminator intermediate layer) loss     (2.0)
    c_fm: float = 2.0
    # Weight of the KL-divergence loss from the posterior encoder                (1.0)
    c_kl: float = 1.0
    # DataLoader worker subprocesses for parallel audio loading                  (4)
    num_workers: int = 4
    # Mixed precision mode:
    #   "fp16"  — half precision, uses GradScaler to prevent underflow (GPU only)
    #   "bf16"  — bfloat16, no GradScaler needed; requires Ampere GPU or newer
    #   "off"   — full fp32 everywhere; always used on CPU automatically         ("fp16")
    amp: str = "fp16"
    # Base RNG seed; each epoch uses seed + epoch so resumed runs reproduce exactly (1234)
    seed: int = 1234
    # Cache STFT spectrograms to disk as float16 to avoid recomputing them       (True)
    cache_specs: bool = True
    # Fraction of clips held out as a validation set (mel loss only, not trained on) (0.05)
    val_fraction: float = 0.05
    # Maximum number of validation clips regardless of val_fraction              (16)
    val_max: int = 16
    # Length of the audio preview clip rendered at each checkpoint export        (3.0 s)
    preview_seconds: float = 3.0
    # EMA smoothing for the live loss chart (0 = raw values); CSV always stores raw (0.6)
    chart_ema: float = 0.6
    # Abort before training if the dataset GPU cache would exceed available VRAM (True)
    check_vram: bool = True
    # Abort before training if the dataset CPU cache would exceed available RAM  (True)
    check_ram: bool = True
    # VRAM reserved for the forward+backward pass after dataset cache            (512 MiB)
    vram_headroom_mb: int = 512
    # RAM reserved for the OS and app after dataset cache                        (4096 MiB)
    ram_headroom_mb: int = 4096
    # ! VRAM cost estimate per sample×frame for the activation headroom check    (0.75 MiB)
    vram_activation_mib_per_sample_frame: float = 0.75
    # ! Assumed total parameter count when the model cannot be inspected for VRAM sizing (120_000_000)
    vram_fallback_model_params: int = 120_000_000
    # CPU intra-op thread count; 0 = auto-detect physical cores                  (0)
    cpu_threads: int = 0
    # Compile the model with torch.compile on CPU training runs                  (True)
    cpu_compile: bool = True
    # torch.compile mode for CPU training:
    #   "default"          — balanced; picks safe kernel strategies, moderate compile time
    #   "reduce-overhead"  — one compile per unique tensor shape, no kernel benchmarking;
    #                        best choice for CPU: same runtime speed as max-autotune but
    #                        compile time drops from ~30 min to ~3-5 min because it does
    #                        not spawn hundreds of cl.exe processes to benchmark kernel variants
    #   "max-autotune"     — benchmarks every kernel shape to find the fastest variant;
    #                        worthwhile on CUDA (picks among cuBLAS, Triton, etc.) but on
    #                        CPU the only compiler is cl.exe, so the benchmark is just wasted
    #                        process-spawn overhead with no runtime benefit
    #   "max-autotune-no-cudagraphs" — manually same as max-autotune on CPU; cudagraphs are
    #                                  a CUDA-only feature so the "no-cudagraphs" qualifier
    #                                  changes nothing when running on CPU
    cpu_compile_mode: str = "reduce-overhead"
    # Batch size per gradient-accumulation step (actual batch = batches × grad_acc) (4)
    batches: int = 4
    # Save a checkpoint and export a .pth every this many epochs                (10)
    frequency: int = 10
    # Total epochs to train before stopping                                      (50)
    epochs: int = 50


# --------------------------------------------------------------------------- #
# Task-button UI (progress overlay / pulse / busy message on TaskManager buttons)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UiConfig:
    # RGBA tint drawn over a task button while it is running or pulsing          ((119, 185, 0, 100))
    overlay_color: tuple = (119, 185, 0, 100)
    # Duration of the progress bar fill / unfill animation                       (450 ms)
    progress_anim_ms: int = 450
    # Half-cycle period of the realtime "live" pulse animation                   (800 ms)
    pulse_period_ms: int = 800
    # Alpha at the dim and bright ends of the pulse cycle                        ((60, 160))
    pulse_alpha_range: tuple = (60, 160)
    # How long "Busy with X" stays visible on a button blocked by another task   (3000 ms)
    busy_message_ms: int = 3000


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