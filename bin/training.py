from __future__ import annotations

import os
import time
import random
from typing import Optional
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from bin.config import FEAT_DIM, SRProfile, TrainConfig
from bin.log import get_logger
from lib.audio.bucket_sampler import BucketBatchSampler, clip_lengths_in_frames
from lib.audio.audio_io import load_audio, audio_info
from lib.audio.dsp import compute_spec, mel_filterbank

log = get_logger("training")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)



# --------------------------------------------------------------------------- #
# CPU / AMP helpers
# --------------------------------------------------------------------------- #
_ISA_TOKENS = ("avx2", "avx512f", "avx512_bf16", "avx512bf16", "amx_bf16", "avx512_fp16")


def _cpu_feature_flags() -> set[str]:
    """Best-effort CPU ISA detection without making py-cpuinfo mandatory.

    All three sources are always tried and their results merged, so a partial
    detection from one does not shadow a more complete result from another.
    """
    flags: set[str] = set()

    # 1. PyTorch native — most reliable when available.
    try:
        get_caps = getattr(torch.cpu, "get_capabilities", None)
        if get_caps is not None:
            text = str(get_caps()).lower()
            flags.update(t for t in _ISA_TOKENS if t in text)
    except Exception:
        pass

    # 2. py-cpuinfo — optional but widely installed.
    try:
        import cpuinfo # type: ignore
        flags.update(str(x).lower() for x in (cpuinfo.get_cpu_info().get("flags") or []))
    except Exception:
        pass

    # 3. /proc/cpuinfo — Linux fallback.
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
            text = fh.read().lower()
        flags.update(t for t in _ISA_TOKENS if t in text)
    except Exception:
        pass

    return flags


def cpu_native_bf16_supported() -> bool:
    """
    Return True only when the CPU appears to expose native BF16 hardware.

    We intentionally do NOT enable CPU BF16 merely because PyTorch can execute
    a BF16 tensor. Zen 2 / Ryzen 3000, for example, should stay FP32.
    """
    flags = _cpu_feature_flags()

    return (
        "avx512_bf16" in flags
        or "avx512bf16" in flags
        or "amx_bf16" in flags
    )


def resolve_amp_dtype(device: torch.device, amp: str):
    """
    Resolve the actual training dtype from the config string.

    Accepted values: "fp16", "bf16", "auto", "off" (anything else -> fp32).

    CUDA:  fp16 -> FP16+GradScaler | bf16/auto -> BF16 if supported else FP16 | off -> FP32
    CPU:   bf16/auto -> BF16 only on AVX512-BF16 / AMX hardware  | fp16/off -> FP32
    """
    amp = str(amp).lower().strip()

    if device.type == "cpu":
        if amp in ("bf16", "auto"):
            if cpu_native_bf16_supported():
                return torch.bfloat16

            if amp != "auto":
                log.warning(
                    "CPU BF16 requested, but native AVX512-BF16/AMX-BF16 "
                    "was not detected; using FP32."
                )

        if amp == "fp16":
            log.warning(
                "CPU FP16 training is disabled intentionally; using FP32."
            )

        return None

    if device.type != "cuda":
        return None

    if amp == "fp16":
        return torch.float16

    if amp == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        log.warning(
            "bf16 requested but this GPU does not support it; "
            "falling back to fp16 + GradScaler"
        )
        return torch.float16

    return None


def configure_cpu_runtime(cfg) -> None:
    """
    Configure CPU threading once, before training begins.

    Physical cores are preferred over logical SMT threads because the RVC
    workload is already heavily vectorized/parallelized internally.
    """
    requested = int(getattr(cfg, "cpu_threads", 0) or 0)

    physical = 0
    try:
        import psutil
        physical = int(psutil.cpu_count(logical=False) or 0)
    except Exception:
        pass

    logical = os.cpu_count() or 1

    if requested > 0:
        threads = requested
    elif physical > 0:
        threads = physical
    else:
        threads = logical

    threads = max(1, min(threads, logical))

    current_threads = torch.get_num_threads()

    if current_threads != threads:
        try:
            torch.set_num_threads(threads)
        except RuntimeError as exc:
            log.warning("Could not set PyTorch CPU thread count: %s", exc)

    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # already initialized by another call site; value is fixed for process lifetime

    log.info(
        "CPU runtime: %d compute threads / 1 inter-op thread "
        "(physical=%d logical=%d)",
        torch.get_num_threads(),
        physical or -1,
        logical,
    )


def compile_cpu_training_models(net_g, net_d, cfg, device: torch.device):
    """
    Compile only the neural-network forward paths.

    We deliberately compile .forward rather than replacing the entire Module.
    That keeps:
      - state_dict keys stable
      - net_g.infer() available
      - checkpoint save/load behavior sane

    dynamic=None lets TorchDynamo specialize first and automatically generalize
    when bucket sequence lengths vary.
    """
    if device.type != "cpu":
        return net_g, net_d

    if not bool(getattr(cfg, "cpu_compile", True)):
        log.info("CPU torch.compile disabled by configuration.")
        return net_g, net_d

    if not hasattr(torch, "compile"):
        log.warning("This PyTorch build has no torch.compile(); using eager CPU.")
        return net_g, net_d

    mode = str(getattr(cfg, "cpu_compile_mode", "max-autotune"))

    try:
        net_g.forward = torch.compile(
            net_g.forward,
            backend="inductor",
            mode=mode,
            dynamic=False,   # specialize on first shape; re-compile on bucket changes
        )

        net_d.forward = torch.compile(
            net_d.forward,
            backend="inductor",
            mode=mode,
            dynamic=False,
        )

        log.info(
            "CPU TorchInductor enabled for G/D: mode=%s dynamic=False",
            mode,
        )

    except Exception as exc:
        log.warning(
            "CPU torch.compile could not be enabled; "
            "falling back to eager execution: %s",
            exc,
        )

    return net_g, net_d


# --------------------------------------------------------------------------- #
# training step
# --------------------------------------------------------------------------- #
class TrainStep:
    def __init__(self, net_g, net_d, optimizer_g, optimizer_d, profile: SRProfile,
                 cfg: TrainConfig, device: torch.device):
        self.net_g, self.net_d = net_g, net_d
        self.optim_g, self.optim_d = optimizer_g, optimizer_d
        self.profile, self.cfg, self.device = profile, cfg, device
        self.hop = profile.hop
        self.grad_acc = cfg.grad_acc
        self.c_mel, self.c_fm, self.c_kl = cfg.c_mel, cfg.c_fm, cfg.c_kl
        self._step_count = 0

        # ---- mixed precision: bf16 needs no loss scaling; fp16 keeps a GradScaler -----------
        self.amp_dtype = resolve_amp_dtype(device, cfg.amp)

        use_scaler = (
            device.type == "cuda"
            and self.amp_dtype == torch.float16
        )

        self.scaler_g = GradScaler(
            "cuda",
            enabled=use_scaler,
        )
        self.scaler_d = GradScaler(
            "cuda",
            enabled=use_scaler,
        )

        precision_name = {
            None: "fp32",
            torch.bfloat16: "bf16",
            torch.float16: "fp16+GradScaler",
        }.get(self.amp_dtype, str(self.amp_dtype))

        log.info("Training precision: %s", precision_name)

        # Per-instance (a class-level cache would go stale when the sample rate changes)
        self.win_size = profile.filter_length
        self.hann = torch.hann_window(self.win_size, device=device)
        self.mel_basis = torch.from_numpy(
            mel_filterbank(profile.sr, self.win_size, profile.n_mels, 0.0, profile.sr / 2)).to(device)

    @property
    def opt_steps(self) -> int:
        """Optimizer updates so far (a batch is one micro-step; grad_acc micro-steps make one update)."""
        return self._step_count // self.grad_acc

    def _autocast(self):
        if self.amp_dtype is None:
            return nullcontext()

        return autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=True,
        )

    # ---- losses ---- #
    @staticmethod
    def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
        """KL divergence in fp32 (independent of the autocast dtype)."""
        z_p, logs_q, m_p, logs_p, z_mask = (t.float() for t in (z_p, logs_q, m_p, logs_p, z_mask))
        kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
        return torch.sum(kl * z_mask) / (torch.sum(z_mask) + 1e-6)

    @staticmethod
    def feature_loss(fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                t = min(rl.shape[-1], gl.shape[-1])
                loss = loss + F.l1_loss(rl[..., :t], gl[..., :t])
        return loss

    @staticmethod
    def generator_loss(outs):
        return sum(torch.mean((1 - dg) ** 2) for dg in outs)

    @staticmethod
    def discriminator_loss(real, fake):
        return sum(torch.mean((1 - dr) ** 2) for dr in real) + sum(torch.mean(dg ** 2) for dg in fake)

    def mel_spectrogram(self, y):
        pad = (self.win_size - self.hop) // 2
        pad_mode = "reflect" if y.shape[-1] > pad else "constant"

        y = F.pad(y.float().unsqueeze(1), (pad, pad), mode=pad_mode).squeeze(1)

        with autocast(self.device.type, enabled=False):
            spec = torch.stft(
                y,
                self.win_size,
                self.hop,
                self.win_size,
                window=self.hann,
                center=False,
                return_complex=True,
            ).abs()

            mel = torch.log(torch.clamp(torch.matmul(self.mel_basis, spec), min=1e-5))

        return mel.half() if self.amp_dtype == torch.float16 else mel


    def spec_to_mel(self, spec: torch.Tensor) -> torch.Tensor:
        with autocast(self.device.type, enabled=False):
            mel = torch.matmul(self.mel_basis, spec.float())
            mel = torch.log(torch.clamp(mel, min=1e-5))

        return mel.half() if self.amp_dtype == torch.float16 else mel


    @staticmethod
    def slice_segments(x, ids_str, segment_size):
        b, c, t = x.shape
        idx = ids_str.view(-1, 1) + torch.arange(segment_size, device=x.device).view(1, -1)
        valid = ((idx >= 0) & (idx < t)).unsqueeze(1)
        clamped_idx = idx.clamp(min=0, max=max(0, t - 1))
        out = x.gather(2, clamped_idx.unsqueeze(1).expand(-1, c, -1))
        return out * valid


    def __call__(self, batch):
        self._step_count += 1
        do_step = self._step_count % self.grad_acc == 0
        phone, phone_lengths, pitch, pitchf, y, mel_raw, y_lengths, sid, wav_raw = batch

        if self._step_count == 1:
            log.info(
                "First batch: phone=%s y=%s mel=%s wav_raw=%s -> max_frames=%d (%.2fs), batch=%d",
                tuple(phone.shape),
                tuple(y.shape),
                tuple(mel_raw.shape),
                tuple(wav_raw.shape),
                phone.shape[1],
                phone.shape[1] / 100.0,
                phone.shape[0],
            )
            if self.device.type == "cuda":
                log.info(
                    "VRAM before this batch: allocated=%.0f MiB reserved=%.0f MiB",
                    torch.cuda.memory_allocated(self.device) / 2**20,
                    torch.cuda.memory_reserved(self.device) / 2**20,
                )

        t0 = time.perf_counter()
        with self._autocast():
            y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = self.net_g(
                phone, phone_lengths, pitch, pitchf, y, y_lengths, sid
            )

        t1 = time.perf_counter()
        n_samples = y_hat.size(2)
        n_frames = max(1, n_samples // self.hop)
        wav = self.slice_segments(wav_raw, ids_slice * self.hop, n_samples)

        with self._autocast():
            d_r, d_g, _, _ = self.net_d(wav, y_hat.detach())
            loss_d = self.discriminator_loss(d_r, d_g)

        t2 = time.perf_counter()

        self.scaler_d.scale(loss_d / self.grad_acc).backward()

        t3 = time.perf_counter()

        if do_step:
            self.scaler_d.step(self.optim_d)
            self.scaler_d.update()
            self.optim_d.zero_grad(set_to_none=True)

        y_mel = self.slice_segments(mel_raw, ids_slice, n_frames)

        _set_requires_grad(self.net_d, False)
        try:
            with self._autocast():
                d_r, d_g, fmap_r, fmap_g = self.net_d(wav, y_hat)
                y_hat_mel = self.mel_spectrogram(y_hat.squeeze(1))

                mel_frames = min(y_mel.size(-1), y_hat_mel.size(-1), n_frames)
                y_mel_loss = y_mel[..., :mel_frames]
                y_hat_mel_loss = y_hat_mel[..., :mel_frames]

                loss_gen = self.generator_loss(d_g)
                loss_fm = self.feature_loss(fmap_r, fmap_g) * self.c_fm
                loss_mel = F.l1_loss(y_mel_loss, y_hat_mel_loss) * self.c_mel
                loss_kl = self.kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.c_kl
                loss_g = loss_gen + loss_fm + loss_mel + loss_kl

            self.scaler_g.scale(loss_g / self.grad_acc).backward()
        finally:
            _set_requires_grad(self.net_d, True)

        if do_step:
            self.scaler_g.step(self.optim_g)
            self.scaler_g.update()
            self.optim_g.zero_grad(set_to_none=True)

        return {
            "g": loss_g.detach().float(),
            "d": loss_d.detach().float(),
            "mel": loss_mel.detach().float(),
            "fm": loss_fm.detach().float(),
            "kl": loss_kl.detach().float(),
        }


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def split_names(main_dir, fraction: float, max_val: int, seed: int):
    """Deterministic (train, validation) split of the clip names in gt_wavs. Fewer than 5 clips
    -> nothing is held out."""
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(main_dir, "gt_wavs")) if f.endswith(".wav"))
    n_val = 0 if len(names) < 5 else min(max_val, max(1, round(len(names) * fraction)))
    shuffled = names[:]
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


class RVCDataset(Dataset):
    def __init__(self, main_dir, profile: SRProfile, names=None, cache: bool = True):
        self.gt_dir = os.path.join(main_dir, "gt_wavs")
        self.f0_dir = os.path.join(main_dir, "f0")
        self.f0c_dir = os.path.join(main_dir, "f0_coarse")
        self.feat_dir = os.path.join(main_dir, "features")
        self.spec_dir = os.path.join(main_dir, f"specs_{profile.sr}")
        self.mel_dir = os.path.join(main_dir, f"mels_{profile.sr}")
        self.feat100_dir = os.path.join(main_dir, "features_100")
        self.profile = profile
        self.hop = profile.hop
        self.cache = cache
        self.files = (sorted(names) if names is not None else
                      sorted(os.path.splitext(f)[0] for f in os.listdir(self.gt_dir) if f.endswith(".wav")))
        self._mel_basis = torch.from_numpy(
            mel_filterbank(
                profile.sr,
                profile.filter_length,
                profile.n_mels,
                0.0,
                profile.sr / 2,
            ).astype(np.float32)
        )

        if cache:
            os.makedirs(self.spec_dir, exist_ok=True)
            os.makedirs(self.mel_dir, exist_ok=True)
            os.makedirs(self.feat100_dir, exist_ok=True)
        self._runtime_phone_dtype = torch.float32
        self._gpu_cache = {}
        self._ram_cache = {}
        self._cache_backend = "disk"

    def set_runtime_dtype(self, dtype: Optional[torch.dtype]) -> None:
        self._runtime_phone_dtype = dtype or torch.float32

    def dataset_cache_estimate_bytes(self, runtime_phone_dtype=torch.float32) -> int:
        """
        Exact-ish tensor memory estimate for the representation actually kept
        in RAM/VRAM.

        Persistent files remain FP32/int64; only the runtime phone/features tensor
        follows the AMP dtype.
        """
        total = 0

        phone_itemsize = torch.tensor([], dtype=runtime_phone_dtype).element_size()

        for name in self.files:
            # --------------------------------------------------------------- #
            # ContentVec / HuBERT features
            # Stored: [T50, 768] FP32
            # Runtime: repeated 2x -> [T100, 768]
            # --------------------------------------------------------------- #
            feat_path = os.path.join(self.feat_dir, f"{name}.npy")
            feat = np.load(feat_path, mmap_mode="r", allow_pickle=False)

            total += (
                int(feat.shape[0])
                * 2
                * int(feat.shape[1])
                * phone_itemsize
            )

            del feat

            # --------------------------------------------------------------- #
            # Fine F0: FP32
            # --------------------------------------------------------------- #
            f0_path = os.path.join(self.f0_dir, f"{name}.wav.npy")
            f0 = np.load(f0_path, mmap_mode="r", allow_pickle=False)
            total += int(f0.size) * 4
            del f0

            # --------------------------------------------------------------- #
            # Coarse F0: int64
            # --------------------------------------------------------------- #
            f0c_path = os.path.join(self.f0c_dir, f"{name}.wav.npy")
            f0c = np.load(f0c_path, mmap_mode="r", allow_pickle=False)
            total += int(f0c.size) * 8
            del f0c

            # --------------------------------------------------------------- #
            # WAV -> runtime float32 mono
            # --------------------------------------------------------------- #
            info = audio_info(os.path.join(self.gt_dir, f"{name}.wav"))

            samples = int(
                round(info.frames * self.profile.sr / info.samplerate)
            )

            total += samples * 4

            # --------------------------------------------------------------- #
            # Linear spectrogram -> runtime FP32
            #
            # compute_spec() pads by (n_fft-hop)/2 and then uses center=False.
            # For these preprocessed clips this lines up with approximately
            # samples / hop frames.
            # --------------------------------------------------------------- #
            n_fft = self.profile.filter_length
            hop = self.profile.hop
            pad = (n_fft - hop) // 2

            padded = samples + 2 * pad

            spec_frames = max(
                1,
                1 + (padded - n_fft) // hop,
            )

            total += (
                self.profile.spec_channels
                * spec_frames
                * 4
            )

            # --------------------------------------------------------------- #
            # Cached log-mel spectrogram -> runtime FP32
            # --------------------------------------------------------------- #
            total += (
                self.profile.n_mels
                * spec_frames
                * 4
            )

        return int(total)

    def load_to_gpu(self, device: torch.device, progress=None) -> int:
        """Load every dataset item into a GPU tensor dict.  Returns bytes actually allocated."""
        self._gpu_cache = {}
        allocated = 0
        for i, name in enumerate(self.files):
            item = self[name]   # normal disk read / compute_spec on CPU
            self._gpu_cache[name] = tuple(
                t.to(device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                for t in item
            )
            # Drop the CPU-side item immediately so its memory (and any
            # intermediate tensors from compute_spec) can be reclaimed before
            # the next file is loaded.  Without this the working tensors from
            # _load_wav / compute_spec accumulate alongside the growing GPU
            # cache and inflate peak VRAM well above the static estimate.
            del item
            allocated += sum(t.nbytes for t in self._gpu_cache[name] if isinstance(t, torch.Tensor))
            if progress and i % max(1, len(self.files) // 20) == 0:
                progress(i, len(self.files), f"GPU cache: {allocated / 2**20:.0f} MB loaded...")
        return allocated

    def load_to_ram(self, progress=None) -> int:
        """Cache the complete dataset in CPU RAM using the selected runtime dtype."""
        self.clear_gpu_cache()
        self._ram_cache = {}

        allocated = 0

        try:
            for i, name in enumerate(self.files):
                item = self[name]

                self._ram_cache[name] = tuple(
                    t.contiguous() if isinstance(t, torch.Tensor) else t
                    for t in item
                )

                allocated += sum(
                    t.nbytes
                    for t in self._ram_cache[name]
                    if isinstance(t, torch.Tensor)
                )

                if progress and (
                    i % max(1, len(self.files) // 20) == 0
                ):
                    progress(
                        i,
                        len(self.files),
                        f"RAM cache: {allocated / 2**20:.0f} MB loaded...",
                    )

        except (MemoryError, RuntimeError):
            self.clear_ram_cache()
            raise

        self._cache_backend = "ram"
        return allocated


    def clear_ram_cache(self) -> None:
        self._ram_cache = {}


    def clear_gpu_cache(self) -> None:
        self._gpu_cache = {}


    def clear_cache(self) -> None:
        self.clear_gpu_cache()
        self.clear_ram_cache()
        self._cache_backend = "disk"


    def __len__(self):
        return len(self.files)

    def _load_wav(self, path) -> torch.Tensor:
        wav, sr = load_audio(path)
        wav = wav.mean(dim=0)
        if sr != self.profile.sr:
            wav = AF.resample(wav, sr, self.profile.sr)
        return wav

    def _mel_path(self, name):
        return os.path.join(
            self.mel_dir,
            f"{name}.npy",
        )


    def _mel_cache_fresh(self, name) -> bool:
        p = self._mel_path(name)
        spec = self._spec_path(name)
        wav = os.path.join(
            self.gt_dir,
            f"{name}.wav",
        )

        if not os.path.exists(p):
            return False

        if not os.path.exists(spec) or not os.path.exists(wav):
            return False

        if os.path.getmtime(p) < max(
            os.path.getmtime(spec),
            os.path.getmtime(wav),
        ):
            return False

        try:
            arr = np.load(
                p,
                mmap_mode="r",
                allow_pickle=False,
            )
            dtype_ok = arr.dtype == np.float32
            del arr
            return dtype_ok
        except Exception:
            return False


    def _mel(self, name, spec):
        """
        Load a cached FP32 log-mel spectrogram or build it once.

        spec: [frequency, frames]
        mel:  [n_mels, frames]
        """
        if self.cache and self._mel_cache_fresh(name):
            return torch.from_numpy(
                np.load(
                    self._mel_path(name),
                    allow_pickle=False,
                )
            )

        with torch.no_grad():
            mel = torch.log(
                torch.clamp(
                    torch.matmul(
                        self._mel_basis,
                        spec.float(),
                    ),
                    min=1e-5,
                )
            )

        if self.cache:
            tmp = self._mel_path(name) + ".tmp"

            with open(tmp, "wb") as fh:
                np.save(
                    fh,
                    mel.contiguous().numpy().astype(np.float32),
                )

            os.replace(
                tmp,
                self._mel_path(name),
            )

        return mel

    def _feat100_path(self, name):
        return os.path.join(
            self.feat100_dir,
            f"{name}.npy",
        )


    def _load_features_100(self, name):
        """
        Convert the original 50 FPS ContentVec representation to the 100 FPS
        representation once and persist it.

        Existing source features remain untouched.
        """
        src_path = os.path.join(
            self.feat_dir,
            f"{name}.npy",
        )

        dst_path = self._feat100_path(name)

        if self.cache and os.path.exists(dst_path):
            if os.path.getmtime(dst_path) >= os.path.getmtime(src_path):
                try:
                    arr = np.load(
                        dst_path,
                        mmap_mode="r",
                        allow_pickle=False,
                    )

                    if (
                        arr.ndim == 2
                        and arr.shape[1] == FEAT_DIM
                        and arr.shape[0] >= 2
                    ):
                        return arr
                except Exception:
                    pass

        src = np.load(
            src_path,
            allow_pickle=False,
        )

        expanded = np.repeat(
            src,
            2,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

        if self.cache:
            tmp = dst_path + ".tmp"

            with open(tmp, "wb") as fh:
                np.save(
                    fh,
                    expanded,
                )

            os.replace(
                tmp,
                dst_path,
            )

            return np.load(
                dst_path,
                mmap_mode="r",
                allow_pickle=False,
            )

        return expanded

    def _spec_path(self, name):
        return os.path.join(self.spec_dir, f"{name}.npy")

    def _cache_fresh(self, name) -> bool:
        p = self._spec_path(name)
        wav = os.path.join(self.gt_dir, f"{name}.wav")

        if not os.path.exists(p) or not os.path.exists(wav):
            return False

        if os.path.getmtime(p) < os.path.getmtime(wav):
            return False

        try:
            arr = np.load(p, mmap_mode="r", allow_pickle=False)
            dtype_ok = arr.dtype == np.float32
            del arr
            return dtype_ok
        except Exception:
            return False

    def _spec(self, name, wav_t):
        """Load FP32 cached spectrogram or compute it once."""
        if self.cache and self._cache_fresh(name):
            return torch.from_numpy(
                np.load(
                    self._spec_path(name),
                    allow_pickle=False,
                )
            )

        spec = compute_spec(
            wav_t,
            self.profile,
        )

        if self.cache:
            tmp = self._spec_path(name) + ".tmp"

            with open(tmp, "wb") as fh:
                np.save(
                    fh,
                    spec.contiguous().numpy().astype(np.float32),
                )

            os.replace(
                tmp,
                self._spec_path(name),
            )

        return spec

    def warm_cache(self, progress=None) -> int:
        """
        Build missing linear-spectrogram and mel caches.

        Each WAV is decoded only once during warm-up.
        """
        built = 0

        for i, name in enumerate(self.files):
            wav_t = None

            spec_fresh = (
                self.cache
                and self._cache_fresh(name)
            )

            mel_fresh = (
                self.cache
                and self._mel_cache_fresh(name)
            )

            if not spec_fresh or not mel_fresh:
                wav_t = self._load_wav(
                    os.path.join(
                        self.gt_dir,
                        f"{name}.wav",
                    )
                )

            if not spec_fresh:
                spec = self._spec(
                    name,
                    wav_t,
                )
                built += 1
            else:
                spec = self._spec(
                    name,
                    wav_t,
                ) if not mel_fresh else None

            if not mel_fresh:
                if spec is None:
                    spec = self._spec(
                        name,
                        wav_t,
                    )

                self._mel(
                    name,
                    spec,
                )

            if progress and i % 200 == 0:
                progress(
                    i,
                    len(self.files),
                )

        return built

    def __getitem__(self, idx):
        name = self.files[idx] if isinstance(idx, int) else idx
        if name in self._gpu_cache:
            return self._gpu_cache[name]

        if name in self._ram_cache:
            return self._ram_cache[name]
        wav_t = self._load_wav(os.path.join(self.gt_dir, f"{name}.wav"))
        pitchf = torch.from_numpy(np.load(os.path.join(self.f0_dir, f"{name}.wav.npy"))).float()
        pitch = torch.from_numpy(np.load(os.path.join(self.f0c_dir, f"{name}.wav.npy"))).long()
        # feats = torch.from_numpy(np.repeat(np.load(os.path.join(self.feat_dir, f"{name}.npy")), 2, axis=0))#.float() Optimization for repeating features
        feats = torch.from_numpy(np.asarray(self._load_features_100(name)))

        if feats.dtype != self._runtime_phone_dtype:
            feats = feats.to(self._runtime_phone_dtype)

        wav_t = wav_t.float()
        spec = self._spec(name, wav_t).float()
        mel = self._mel(name, spec).float()

        n = min(spec.size(1), mel.size(1), feats.size(0), pitchf.size(0), pitch.size(0))
        if n <= 0:
            raise ValueError(f"Dataset item '{name}' has 0 valid frames (spec: {spec.size(1)}, mel: {mel.size(1)} feats: {feats.size(0)}, pitch: {pitch.size(0)})")
        return feats[:n], pitch[:n], pitchf[:n], spec[:, :n], mel[:, :n], wav_t[: n * self.hop], 0


class RVCCollate:
    def __init__(
        self,
        profile: SRProfile,
        max_frames: Optional[int] = 500,
        device: Optional[torch.device] = None,
    ):
        self.hop = profile.hop
        self.spec_ch = profile.spec_channels
        self.n_mels = profile.n_mels
        self.max_frames = max_frames
        self.device = device or torch.device("cpu")

        self._capacity_b = 0
        self._capacity_t = 0
        self._phone_dtype = None
        self._buffers = None

    def _ensure_buffers(
        self,
        b: int,
        t: int,
        phone_dtype: torch.dtype,
    ):
        needs_new = (
            self._buffers is None
            or self._capacity_b < b
            or self._capacity_t < t
            or self._phone_dtype != phone_dtype
        )

        if not needs_new:
            return

        cap_b = max(
            b,
            self._capacity_b,
        )

        cap_t = max(
            t,
            self._capacity_t,
        )

        device = self.device

        self._buffers = {
            "phone": torch.empty(
                cap_b,
                cap_t,
                FEAT_DIM,
                dtype=phone_dtype,
                device=device,
            ),
            "pitch": torch.empty(
                cap_b,
                cap_t,
                dtype=torch.long,
                device=device,
            ),
            "pitchf": torch.empty(
                cap_b,
                cap_t,
                dtype=torch.float32,
                device=device,
            ),
            "spec": torch.empty(
                cap_b,
                self.spec_ch,
                cap_t,
                dtype=torch.float32,
                device=device,
            ),
            "mel": torch.empty(
                cap_b,
                self.n_mels,
                cap_t,
                dtype=torch.float32,
                device=device,
            ),
            "wav": torch.empty(
                cap_b,
                1,
                cap_t * self.hop,
                dtype=torch.float32,
                device=device,
            ),
            "sid": torch.empty(
                cap_b,
                dtype=torch.long,
                device=device,
            ),
        }

        self._capacity_b = cap_b
        self._capacity_t = cap_t
        self._phone_dtype = phone_dtype

    def __call__(self, batch):
        if self.max_frames:
            batch = [
                (
                    f[:self.max_frames],
                    p[:self.max_frames],
                    pf[:self.max_frames],
                    s[:, :self.max_frames],
                    m[:, :self.max_frames],
                    w[:self.max_frames * self.hop],
                    sid,
                )
                for f, p, pf, s, m, w, sid in batch
            ]

        batch = sorted(
            batch,
            key=lambda x: x[0].size(0),
            reverse=True,
        )

        lengths = torch.as_tensor(
            [x[0].size(0) for x in batch],
            dtype=torch.long,
            device=self.device,
        )

        b = len(batch)
        max_frames = int(lengths.max().item())

        phone_dtype = batch[0][0].dtype

        self._ensure_buffers(
            b,
            max_frames,
            phone_dtype,
        )

        phone = self._buffers["phone"][
            :b,
            :max_frames,
        ]

        pitch = self._buffers["pitch"][
            :b,
            :max_frames,
        ]

        pitchf = self._buffers["pitchf"][
            :b,
            :max_frames,
        ]

        spec = self._buffers["spec"][
            :b,
            :,
            :max_frames,
        ]

        mel = self._buffers["mel"][
            :b,
            :,
            :max_frames,
        ]

        wav = self._buffers["wav"][
            :b,
            :,
            :max_frames * self.hop,
        ]

        sid = self._buffers["sid"][:b]

        for i, (
            f,
            p,
            pf,
            s,
            m,
            w,
            speaker,
        ) in enumerate(batch):
            n = f.size(0)

            phone[i, :n].copy_(f)
            pitch[i, :n].copy_(p)
            pitchf[i, :n].copy_(pf)
            spec[i, :, :n].copy_(s)
            mel[i, :, :n].copy_(m)
            wav[i, 0, :w.size(0)].copy_(w)
            sid[i] = speaker

            # Only clear padding.
            # The valid regions are overwritten every batch.
            if n < max_frames:
                phone[i, n:max_frames].zero_()
                pitch[i, n:max_frames].zero_()
                pitchf[i, n:max_frames].zero_()
                spec[i, :, n:max_frames].zero_()
                mel[i, :, n:max_frames].zero_()

            wav_len = w.size(0)
            total_wav = max_frames * self.hop

            if wav_len < total_wav:
                wav[i, 0, wav_len:total_wav].zero_()

        return (
            phone,
            lengths,
            pitch,
            pitchf,
            spec,
            mel,
            lengths,
            sid,
            wav,
        )


def create_dataloader(
    main_dir,
    profile: SRProfile,
    batch_size: int,
    num_workers: int,
    names=None,
    cache: bool = True,
    seed: int = 1234,
    device: Optional[torch.device] = None,
    amp_dtype=None,
    vram_headroom_mb=1500,
    ram_headroom_mb=4096,
    max_frames: int = 500,
    check_vram: bool = True,
    check_ram: bool = True,
    model_params: Optional[int] = None,
    vram_activation_mib_per_sample_frame: float = 0.75,
    vram_fallback_model_params: int = 120_000_000,
) -> DataLoader:
    dataset = RVCDataset(main_dir, profile, names, cache)
    lengths = clip_lengths_in_frames(dataset.gt_dir, dataset.files, profile.hop)
    sampler = BucketBatchSampler(lengths, batch_size, seed=seed)

    if cache:
        built = dataset.warm_cache()
        if built:
            log.info("Built %d FP32 spectrogram cache file(s)", built)

    _device = device or torch.device("cpu")
    training_vram_est = 0

    if _device.type == "cuda" and check_vram:
        training_vram_est = estimate_training_vram_bytes(
            profile=profile,
            batch_size=batch_size,
            max_frames=max_frames,
            amp_dtype=amp_dtype,
            model_params=model_params,
            activation_mib_per_sample_frame=vram_activation_mib_per_sample_frame,
            fallback_model_params=vram_fallback_model_params,
        )

        check_vram_feasibility(
            device=_device,
            training_vram_estimate=training_vram_est,
            vram_headroom_mb=vram_headroom_mb,
            batch_size=batch_size,
        )
    elif _device.type == "cuda":
        log.warning("VRAM pre-flight check disabled by configuration.")

    backend = prepare_dataset_cache(
        dataset,
        _device,
        amp_dtype,
        vram_headroom_mb,
        ram_headroom_mb,
        training_vram_estimate=training_vram_est,
        check_vram=check_vram,
        check_ram=check_ram,
    )

    if _device.type == "cpu" or backend == "gpu":
        safe_workers = 0
    else:
        import platform
        safe_workers = 0 if platform.system() == "Windows" else num_workers

    collate = RVCCollate(
        profile,
        device=_device if backend == "gpu" else torch.device("cpu"),
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=safe_workers,
        pin_memory=_device.type == "cuda" and backend != "gpu",
        collate_fn=collate,
        persistent_workers=safe_workers > 0,
    )

    log.info(
        "Dataset cache backend selected: %s | workers=%d | pin_memory=%s",
        backend,
        safe_workers,
        backend != "gpu",
    )

    return loader


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True

    msg = str(exc).lower()
    return (
        "out of memory" in msg
        or "cuda out of memory" in msg
        or "cublas_status_alloc_failed" in msg
    )


# --------------------------------------------------------------------------- #
# VRAM estimation
# --------------------------------------------------------------------------- #

def count_model_parameters(*modules) -> int:
    """Count trainable parameters across one or more modules."""
    total = 0

    for module in modules:
        if module is None:
            continue

        total += sum(
            p.numel()
            for p in module.parameters()
            if p.requires_grad
        )

    return int(total)


def estimate_training_vram_bytes(
    profile: "SRProfile",
    batch_size: int,
    max_frames: int,
    amp_dtype,
    model_params: Optional[int] = None,
    activation_mib_per_sample_frame: float = 0.75,
    fallback_model_params: int = 120_000_000,
) -> int:
    """
    Estimate the VRAM required for one actual training micro-batch.

    This is deliberately calibrated for the current RVC/VITS architecture.

    Important:
        gradient accumulation does NOT reduce this number.
        batch_size is the actual instantaneous micro-batch size.

    model_params:
        Prefer the actual number of trainable G+D parameters.

    activation_mib_per_sample_frame:
        Calibrated hidden-activation envelope.

        Default 0.75 MiB / (sample * frame) gives a realistic estimate for
        this RVC graph while still scaling strongly with large batch sizes.
    """
    batch_size = max(1, int(batch_size))
    max_frames = max(1, int(max_frames))

    if model_params is None:
        model_params = fallback_model_params

    model_params = max(
        1,
        int(model_params),
    )

    # ------------------------------------------------------------------ #
    # Model + optimizer state
    #
    # Current CUDA training keeps model parameters in FP32:
    #
    #   parameter       4 B
    #   gradient        4 B
    #   Adam exp_avg    4 B
    #   Adam exp_avg_sq 4 B
    #
    # = 16 bytes/parameter
    # ------------------------------------------------------------------ #
    model_state_bytes = model_params * 16

    # ------------------------------------------------------------------ #
    # Input tensors
    # ------------------------------------------------------------------ #
    runtime_feature_dtype = (
        amp_dtype
        if amp_dtype in (
            torch.float16,
            torch.bfloat16,
        )
        else torch.float32
    )

    feature_bytes = (
        batch_size
        * max_frames
        * FEAT_DIM
        * torch.tensor(
            [],
            dtype=runtime_feature_dtype,
        ).element_size()
    )

    spec_bytes = (
        batch_size
        * profile.spec_channels
        * max_frames
        * 4
    )

    mel_bytes = (
        batch_size
        * profile.n_mels
        * max_frames
        * 4
    )

    wav_bytes = (
        batch_size
        * max_frames
        * profile.hop
        * 4
    )

    pitch_bytes = (
        batch_size
        * max_frames
        * 12  # int64 pitch + float32 pitchf
    )

    raw_batch_bytes = (
        feature_bytes
        + spec_bytes
        + mel_bytes
        + wav_bytes
        + pitch_bytes
    )

    # ------------------------------------------------------------------ #
    # Hidden activations
    # ------------------------------------------------------------------ #
    activation_bytes = int(
        batch_size
        * max_frames
        * float(activation_mib_per_sample_frame)
        * 2**20
    )

    # A modest reserve for temporary outputs / allocator fragmentation.
    # The previous ×2 raw-batch reserve was unnecessarily aggressive.
    transient_bytes = int(
        raw_batch_bytes * 1.25
    )

    # Keep this intentionally modest. This is not supposed to swallow an
    # entire gigabyte of VRAM before the real graph has run.
    workspace_bytes = 384 * 2**20

    total = (
        model_state_bytes
        + activation_bytes
        + transient_bytes
        + workspace_bytes
    )

    log.info(
        "Training VRAM estimate: %.0f MiB "
        "(G+D model+Adam=%.0f MiB, activations=%.0f MiB, "
        "batch/transient=%.0f MiB, workspace=%.0f MiB | "
        "params=%.1fM batch=%d max_frames=%d)",
        total / 2**20,
        model_state_bytes / 2**20,
        activation_bytes / 2**20,
        transient_bytes / 2**20,
        workspace_bytes / 2**20,
        model_params / 1_000_000,
        batch_size,
        max_frames,
    )

    return int(total)


def check_vram_feasibility(
    device: torch.device,
    training_vram_estimate: int,
    vram_headroom_mb: int,
    batch_size: int,
) -> None:
    """
    Reject a micro-batch only when the estimated training footprint cannot fit.

    Uses the stricter of:
      - CUDA driver's currently free memory
      - memory not already owned by this PyTorch process
    """
    if device.type != "cuda":
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    allocated_by_pytorch = torch.cuda.memory_allocated(device)

    allocator_free = max(
        0,
        total_bytes - allocated_by_pytorch,
    )

    available_bytes = min(
        free_bytes,
        allocator_free,
    )

    headroom_bytes = (
        max(0, int(vram_headroom_mb))
        * 2**20
    )

    safe_budget = max(
        0,
        available_bytes - headroom_bytes,
    )

    # The estimator includes model memory. If the model is already resident,
    # don't charge those bytes twice.
    required_additional = max(
        0,
        training_vram_estimate - allocated_by_pytorch,
    )

    log.info(
        "VRAM feasibility: batch=%d | estimate=%.0f MiB | "
        "additional required=%.0f MiB | safe budget=%.0f MiB "
        "(driver-free=%.0f MiB allocator-free=%.0f MiB "
        "headroom=%d MiB)",
        batch_size,
        training_vram_estimate / 2**20,
        required_additional / 2**20,
        safe_budget / 2**20,
        free_bytes / 2**20,
        allocator_free / 2**20,
        vram_headroom_mb,
    )

    if required_additional <= safe_budget:
        return

    raise RuntimeError(
        f"Unsafe batch size {batch_size}: estimated additional "
        f"training VRAM is {required_additional / 2**20:.0f} MiB, "
        f"but only {safe_budget / 2**20:.0f} MiB is safely available. "
        f"Reduce batch_size. Gradient accumulation cannot make a "
        f"single oversized micro-batch fit."
    )


def prepare_dataset_cache(
    dataset: RVCDataset,
    device: torch.device,
    amp_dtype,
    vram_headroom_mb: int,
    ram_headroom_mb: int,
    training_vram_estimate: int = 0,
    check_vram: bool = True,
    check_ram: bool = True,
    progress=None,
) -> str:
    phone_dtype = (
        amp_dtype
        if amp_dtype in (torch.float16, torch.bfloat16)
        else torch.float32
    )

    dataset.set_runtime_dtype(phone_dtype)
    estimate = dataset.dataset_cache_estimate_bytes(phone_dtype)

    log.info(
        "Dataset runtime cache estimate: %.0f MB (phone/features=%s, spec/f0/wav=FP32)",
        estimate / 2**20,
        str(phone_dtype).replace("torch.", ""),
    )

    if device.type == "cuda":
        if check_vram:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            allocated_by_pytorch = torch.cuda.memory_allocated(device)
            allocator_free = max(0, total_bytes - allocated_by_pytorch)
            effective_free = min(free_bytes, allocator_free)
            loading_overhead = estimate // 4
            training_reserve = max(0, training_vram_estimate - allocated_by_pytorch)

            vram_budget = max(
                0,
                effective_free
                - vram_headroom_mb * 2**20
                - loading_overhead
                - training_reserve,
            )

            log.info(
                "GPU cache check: %.0f MB required, %.0f MB usable "
                "(%.0f MB effective free - %.0f MB headroom - "
                "%.0f MB load overhead - %.0f MB training reserve)",
                estimate / 2**20,
                vram_budget / 2**20,
                effective_free / 2**20,
                vram_headroom_mb,
                loading_overhead / 2**20,
                training_reserve / 2**20,
            )

            try_gpu_cache = estimate <= vram_budget
        else:
            log.warning(
                "GPU dataset-cache capacity check disabled; attempting VRAM cache directly."
            )
            try_gpu_cache = True

        if try_gpu_cache:
            try:
                allocated = dataset.load_to_gpu(device, progress=progress)
                torch.cuda.synchronize(device)
                free_after = torch.cuda.mem_get_info(device)[0]
                dataset._cache_backend = "gpu"

                log.info(
                    "Dataset cache: GPU | %.0f MB allocated | %.0f MB VRAM remaining",
                    allocated / 2**20,
                    free_after / 2**20,
                )
                return "gpu"

            except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
                log.warning(
                    "GPU dataset cache allocation failed; falling back to RAM: %s",
                    exc,
                )
                dataset.clear_gpu_cache()
                torch.cuda.empty_cache()

            except RuntimeError as exc:
                if not _is_cuda_oom(exc):
                    raise

                log.warning(
                    "GPU dataset cache hit CUDA OOM; falling back to RAM: %s",
                    exc,
                )
                dataset.clear_gpu_cache()
                torch.cuda.empty_cache()

    def _available_ram_bytes() -> int:
        try:
            import psutil
            return int(psutil.virtual_memory().available)
        except Exception:
            pass

        if hasattr(os, "sysconf"):
            try:
                return int(
                    os.sysconf("SC_AVPHYS_PAGES")
                    * os.sysconf("SC_PAGE_SIZE")
                )
            except Exception:
                pass

        if os.name == "nt":
            try:
                import ctypes

                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                status = MEMORYSTATUSEX()
                status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)

                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                    return int(status.ullAvailPhys)
            except Exception:
                pass

        return 0

    if not check_ram:
        log.warning("RAM capacity check disabled; attempting full RAM dataset cache.")

        try:
            allocated = dataset.load_to_ram(progress=progress)
            log.info(
                "Dataset cache: RAM | %.0f MB allocated (RAM pre-flight check disabled)",
                allocated / 2**20,
            )
            return "ram"
        except (MemoryError, RuntimeError) as exc:
            log.warning(
                "RAM dataset cache allocation failed; falling back to disk: %s",
                exc,
            )
            dataset.clear_ram_cache()
            return "disk"

    available_ram = _available_ram_bytes()
    ram_budget = max(0, available_ram - ram_headroom_mb * 2**20)

    log.info(
        "RAM cache check: %.0f MB required, %.0f MB usable "
        "(%.0f MB available - %.0f MB headroom)",
        estimate / 2**20,
        ram_budget / 2**20,
        available_ram / 2**20,
        ram_headroom_mb,
    )

    if available_ram > 0 and estimate <= ram_budget:
        try:
            allocated = dataset.load_to_ram(progress=progress)
            log.info("Dataset cache: RAM | %.0f MB allocated", allocated / 2**20)
            return "ram"
        except (MemoryError, RuntimeError) as exc:
            log.warning(
                "RAM dataset cache allocation failed; falling back to disk: %s",
                exc,
            )
            dataset.clear_ram_cache()

    dataset.clear_cache()

    log.info(
        "Dataset cache: DISK | %.0f MB required but insufficient safe memory was available",
        estimate / 2**20,
    )
    return "disk"


# --------------------------------------------------------------------------- #
# Validation / listening previews
# --------------------------------------------------------------------------- #
@torch.no_grad()
def synthesize(step: TrainStep, net_g, item, max_frames: int | None = None):
    feats, pitch, pitchf, spec, mel, wav, _sid = item

    n = feats.size(0) if max_frames is None else min(feats.size(0), max_frames)
    dev = step.device

    p_len = torch.full((1,), n, dtype=torch.long, device=dev)

    with step._autocast():
        out = net_g.infer(
            feats[:n].unsqueeze(0).to(dev),
            p_len,
            pitch[:n].unsqueeze(0).to(dev),
            pitchf[:n].unsqueeze(0).to(dev),
            torch.zeros(1, dtype=torch.long, device=dev),
        )

    if isinstance(out, tuple):
        out = out[0]

    pred = out.reshape(-1).float()
    gt = wav[:n * step.hop].to(dev)
    gt_mel = mel[:, :n].to(dev)

    m = min(pred.numel(), gt.numel())
    return pred[:m], gt[:m], gt_mel


@torch.no_grad()
def validate(step: TrainStep, net_g, dataset: RVCDataset, max_frames: int = 600) -> float:
    net_g.eval()

    try:
        errs = []

        for i in range(len(dataset)):
            pred, gt, gt_mel = synthesize(step, net_g, dataset[i], max_frames)
            pred_mel = step.mel_spectrogram(pred.unsqueeze(0))

            t = min(gt_mel.shape[-1], pred_mel.shape[-1])

            errs.append(
                F.l1_loss(
                    gt_mel.unsqueeze(0)[..., :t],
                    pred_mel[..., :t],
                ).item()
            )

        return float(np.mean(errs)) if errs else float("nan")
    finally:
        net_g.train()