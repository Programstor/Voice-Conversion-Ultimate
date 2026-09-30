"""
bin/modules.py - orchestration: model loading, offline inference, realtime, dataset
preprocessing, training and index building.

The heavy lifting lives elsewhere:
    bin.pipeline  - features, retrieval, conversion, realtime engine (sample-rate agnostic)
    bin.training  - train step / dataset / loader
    bin.config    - every constant (paths, sample-rate profiles, DSP, realtime, index, training)
    lib.audio.dsp - all DSP / filter building blocks
    lib.audio.audio_io - audio load / save (soundfile-backed, no FFmpeg dependency)
"""
from __future__ import annotations

import csv
import datetime
import gc
import math
import os
import pickle
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional
from functools import wraps
from wakepy import keep

import faiss
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from bin.config import AppConfig, FEAT_DIM, FEAT_SR, SRProfile, TaskCancelled, get_profile
from lib.audio.dsp import zero_phase_highpass, stereo_diffusion
from bin.log import get_logger
from bin.pipeline import IndexBank, Pipeline, RealtimeEngine, load_index_bank
from bin.training import (RVCDataset, TrainStep, resolve_amp_dtype, create_dataloader,
                          seed_everything, split_names,synthesize, validate, 
                          configure_cpu_runtime, compile_cpu_training_models)
from lib.audio.audio_io import load_audio, save_audio, audio_info
from lib.audio.models import SynthesizerTrnMs768NSFsid
from lib.audio.slicer import Slicer
from lib.ui.lslider import LiveParams

try:  # v2 layout (8 periods) matches the pretrained f0D*.pth files
    from lib.audio.models import MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator
except ImportError:  # pragma: no cover
    from lib.audio.models import MultiPeriodDiscriminator  # type: ignore

log = get_logger("modules")

torch.set_float32_matmul_precision("high")   # TF32 matmuls; cudnn convs already use TF32 by default
torch.backends.cudnn.benchmark = False

_KEYS = ("g", "d", "mel", "fm", "kl")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def keep_awake(func):
    """Decorator to prevent system sleep while a method or function runs."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with keep.running():
            return func(*args, **kwargs)
    return wrapper

def load_checkpoint(path, map_location) -> Any:
    """torch.load with weights_only=True. Checkpoints that need arbitrary pickled objects are
    refused unless VCU_ALLOW_UNSAFE_LOAD=1 is set (loading them can execute code)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and "weights_only" not in str(exc):
            raise
        if os.environ.get("VCU_ALLOW_UNSAFE_LOAD") == "1":
            log.warning("Loading %s with full pickle support (VCU_ALLOW_UNSAFE_LOAD=1)", path)
            return torch.load(path, map_location=map_location, weights_only=False)
        raise RuntimeError(
            f"{path} contains non-tensor Python objects and was refused for safety. If you trust "
            f"the file, set VCU_ALLOW_UNSAFE_LOAD=1 and try again.") from exc


def load_state_logged(module: torch.nn.Module, state: dict, name: str, *,
                      raise_missing: bool = True, raise_unexpected: bool = False) -> None:
    """load_state_dict(strict=False) that reports what did not match instead of hiding it."""
    res = module.load_state_dict(state, strict=False)
    if res.missing_keys:
        log.warning("%s: %d missing keys, e.g. %s", name, len(res.missing_keys), res.missing_keys[:3])
    if res.unexpected_keys:
        log.warning("%s: %d unexpected keys, e.g. %s", name, len(res.unexpected_keys), res.unexpected_keys[:3])
    if (res.missing_keys and raise_missing) or (res.unexpected_keys and raise_unexpected):
        raise RuntimeError(f"{name}: checkpoint does not match the model "
                           f"({len(res.missing_keys)} missing, {len(res.unexpected_keys)} unexpected keys)")


def _to_sr(value) -> int:
    if isinstance(value, str):
        v = value.strip().lower()
        return int(float(v[:-1]) * 1000) if v.endswith("k") else int(v)
    return int(value)


def _file_sig(path: Path):
    st = path.stat()
    return (str(path), st.st_mtime_ns, st.st_size)


def _require_model_name(mo_name: str) -> str:
    """Guard against empty/whitespace-only model names, which would otherwise resolve to
    the main training folder itself (self.paths.training_run("") == the training root),
    silently scattering gt_wavs/features/f0/etc. straight into it."""
    name = (mo_name or "").strip()
    if not name:
        raise ValueError("Write a valid name for the model")
    return name


def _is_sticky_cuda_error(exc: BaseException) -> bool:
    """True for an unrecoverable ("sticky") CUDA context, e.g. after a device-side assert.
    Once this happens, EVERY further CUDA call in this process fails the same way -- there is
    no way to recover from user code, only a process restart. See:
    https://docs.pytorch.org/docs/stable/notes/cuda.html (device-side assertions)."""
    msg = str(exc)
    return "device-side assert" in msg or "CUDA error" in msg or "AcceleratorError" in type(exc).__name__


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            if _is_sticky_cuda_error(exc):
                # A previous crash already poisoned this process's CUDA context; empty_cache()
                # touching CUDA again just re-raises the same sticky error. Log it plainly once
                # instead of letting it look like a new, unrelated crash on every button click.
                log.error("CUDA context is unrecoverable after an earlier crash (%s). "
                         "Restart the application to continue using the GPU.", exc)
            else:
                raise


# --------------------------------------------------------------------------- #
class Modules:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.paths = cfg.paths
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_half = self.device == "cuda"

        self.pipeline = Pipeline(cfg, self.device, self.is_half)
        self.realtime = RealtimeEngine(cfg, self.pipeline)

        self.net_g = None
        self.bank: Optional[IndexBank] = None
        self.tgt_sr: Optional[int] = None
        self._model_sig = None
        self._index_sig = None
        self.current_model_path: Optional[str] = None
        self.current_index_path: Optional[str] = None

        self._state_lock = threading.RLock()
        self._training_active = False

    # ------------------------------------------------------------------ loading
    def _build_generator(self, path: Path, strip_weight_norm: bool = True):
        cpt = load_checkpoint(path, "cpu")
        if (not isinstance(cpt, dict)
            or "config" not in cpt
            or str(cpt.get("version", "v2")) != "v2"
            or int(cpt.get("f0", 1)) != 1):
            raise ValueError("Invalid RVC v2 model")
        
        conf, sr = list(cpt["config"]), _to_sr(list(cpt["config"])[-1])

        if math.prod(conf[12]) != sr // 100:
            raise ValueError("Inconsistent upsample rates")
        
        weights = {k: v for k, v in cpt.get("weight", cpt).items() if not k.startswith("enc_q.")}
        emb_w = weights.get("enc_p.emb_phone.weight")

        if emb_w is not None and emb_w.shape[1] != FEAT_DIM:
            raise ValueError(f"Dim mismatch: {emb_w.shape[1]} != {FEAT_DIM}")

        net_g = SynthesizerTrnMs768NSFsid(*conf, is_half=self.is_half)
        del net_g.enc_q
        load_state_logged(net_g, weights, "generator")

        if strip_weight_norm:
            try:
                for part in (getattr(net_g, "dec", None), getattr(net_g, "flow", None)):
                    if fn := getattr(part, "remove_weight_norm", None):
                        fn()
            except Exception as exc:
                log.warning("remove_weight_norm failed (%s)", exc)
                return self._build_generator(path, strip_weight_norm=False)

        net_g = net_g.eval()
        net_g = net_g.compile_for_inference()
        return ((net_g.half() if self.is_half else net_g.float()).to(self.device), sr, f"{cpt.get('version', 'v2')}: {cpt.get('info', '')}")

    def load_model(self, gen_path, idx_path, progress_callback=None):
        say = lambda i, msg: progress_callback(i, 5, msg) if progress_callback else None
        say(0, "Started model loading...")
        if self._training_active:
            say(0, "Can't load a voice while training")
            return False

        with self._state_lock:
            abs_model = self.paths.models / gen_path if gen_path else None
            if not abs_model or not abs_model.is_file():
                say(0, f"Missing model: {abs_model}")
                return False

            sig = _file_sig(abs_model)
            if self.net_g is not None and sig == self._model_sig:
                say(2, f"Model already loaded: {gen_path}")
            else:
                try:
                    say(1, f"Loading model from {abs_model}...")
                    self.net_g, self.tgt_sr, info = self._build_generator(abs_model)
                    self._model_sig, self.current_model_path = sig, gen_path
                    say(2, f"Loaded {info}")
                except Exception as exc:
                    self.unload_model()
                    say(0, f"Error loading model: {exc}")
                    return False

            abs_idx = (self.paths.indices / idx_path if idx_path and "No index file found" not in idx_path else None)

            if abs_idx and abs_idx.is_file():
                isig = _file_sig(abs_idx)
                if self.bank is not None and isig == self._index_sig:
                    say(4, f"Index already loaded: {idx_path}")
                else:
                    try:
                        say(3, f"Loading index from {abs_idx}...")
                        bank = load_index_bank(
                            abs_idx,
                            self.device,
                            self.cfg.index.max_bank,
                            self.cfg.index.seed,
                        )
                        if bank.dim != FEAT_DIM:
                            raise ValueError(f"Index dim mismatch: {bank.dim} vs {FEAT_DIM}")
                    
                        self.bank = bank
                        self._index_sig = isig
                        self.current_index_path = idx_path

                        say(4, "Loaded index")
                    except Exception as exc:
                        self.bank, self._index_sig, self.current_index_path = (None, None, None)
                        say(0, f"Error loading index: {exc}")
            else:
                self.bank, self._index_sig, self.current_index_path = (None, None, None)
                say(4, "No index provided!")

        _free_gpu()
        say(5, "Finished model loading!")
        return True

    def unload_model(self, progress_callback=None):
        say = lambda i, msg: progress_callback(i, 3, msg) if progress_callback else None
        say(0, "Started model unloading...")
        with self._state_lock:
            if self.net_g is not None:
                self.net_g.cpu()
            say(0, "Deleted model.")
            self.net_g, self.tgt_sr, self._model_sig, self.current_model_path = (None, None, None, None)
            say(0, "Deleted index.")
            self.bank, self._index_sig, self.current_index_path = (None, None, None)

        _free_gpu()
        say(3, "Finished model unloading!")

    # ---- offline inference ---- #
    @keep_awake
    def infer_audio(self, gen_path, idx_path, audio_path, f0_up_key, vol_scale, protect, index_rate,
                    res_sr, do_instruments=False, keep_stereo=False, progress_callback=None, stop_check=None):
        say = lambda i, msg: progress_callback(i, 6, msg) if progress_callback else None
        try:
            say(0, "Started audio inference...")
            if not self.load_model(gen_path, idx_path):
                raise RuntimeError("Model could not be loaded; see the log for details")
            assert self.net_g is not None and self.tgt_sr is not None
            tgt_sr = self.tgt_sr

            say(1, "Loading audio...")
            wav, orig_sr = load_audio(audio_path)
            is_stereo = wav.shape[0] > 1
            mono = wav.mean(dim=0) if is_stereo else wav.squeeze(0)
            maxv = mono.abs().max().item()
            if maxv > 0.005:
                mono = mono / maxv                     # same scalar — no need to re-reduce wav
            final_sr = res_sr if res_sr > 0 else (44100 if tgt_sr == 40000 else tgt_sr)

            out = self.pipeline.convert_file(
                self.net_g, tgt_sr, mono, orig_sr, f0_up_key=f0_up_key, vol_scale=vol_scale,
                protect=protect, index_rate=index_rate, bank=self.bank, progress_callback=progress_callback,
                stop_check=stop_check)

            say(2, "Final resampling...")
            out = self.pipeline.resample(out.cpu(), tgt_sr, final_sr)

            if keep_stereo and is_stereo:
                say(3, "Adding stereo effects...")
                out = stereo_diffusion(out, final_sr)

            if do_instruments:
                say(4, "Stitching instruments...")
                ip = audio_path.replace("Vocals", "Instrumental")
                if ip != audio_path and os.path.exists(ip):
                    inst, ir = load_audio(ip)
                    inst = self.pipeline.resample(inst.float().cpu(), ir, final_sr)
                    if out.ndim == 1: out = out.unsqueeze(0)
                    if inst.ndim == 1: inst = inst.unsqueeze(0)
                    if out.shape[0] == 1 and inst.shape[0] == 2: out = out.repeat(2, 1)
                    elif out.shape[0] == 2 and inst.shape[0] == 1: inst = inst.repeat(2, 1)
                    length = max(out.shape[-1], inst.shape[-1])
                    out = F.pad(out, (0, length - out.shape[-1]))
                    inst = F.pad(inst, (0, length - inst.shape[-1]))
                    out = out * 1.5 + inst
                else:
                    log.warning("No matching instrumental found for %s", audio_path)

            say(5, "Saving audio...")
            model_name = os.path.splitext(os.path.basename(gen_path))[0]
            song = os.path.splitext(os.path.basename(audio_path))[0]
            match = re.search(r"^(?:\d+_)+(.*)", song)
            song = match.group(1).strip() if match else song.strip()
            song = re.sub(r"[\(\[\{].*?[\)\]\}]", "", song)
            song = re.sub(r"\s+", " ", song).strip().strip("_")

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            ft = "mp3" if final_sr <= 48000 else "wav"
            out_path = str(self.paths.converted / f"{model_name}_{song}_{ts}.{ft}")
            save_audio(out_path, out, int(final_sr))
            say(6, "Finished audio inference!")
            return out_path
        finally:
            _free_gpu()

    # ------------------------------------------------------------------ realtime
    @keep_awake
    def infer_input(self, gen_path, idx_path, live_params: LiveParams, block_ms,
                    mic_device, spk_device, stop_check=None, progress_callback=None):
        in_idx = int(mic_device.split("|")[0].strip())
        out_idx = int(spk_device.split("|")[0].strip())
        try:
            if not self.load_model(gen_path, idx_path, progress_callback):
                raise RuntimeError("Model could not be loaded; see the log for details")
            assert self.net_g is not None and self.tgt_sr is not None
            self.realtime.run(
                self.net_g, self.tgt_sr, self.bank, in_idx, out_idx,
                live_params=live_params, block_ms=int(block_ms),
                stop_check=stop_check, progress_callback=progress_callback)
        finally:
            _free_gpu()

    # ------------------------------------------------------------------ dataset preprocessing
    @keep_awake
    def preprocess_dataset(self, mo_name, ds_path, tgt_sr, progress_callback=None, stop_check=None):
        say = lambda i, n, msg: progress_callback(i, n, msg) if progress_callback else None
        mo_name = _require_model_name(mo_name)
        say(0, 1, "Started dataset preprocess...")

        profile = get_profile(tgt_sr)          # raises for unsupported rates (e.g. 32 kHz)
        hop, dsp = profile.hop, self.cfg.dsp
        alpha, max_perc = 0.75, 0.9
        min_frames = profile.segment_size // hop + 1     # shorter clips cannot be sliced for training

        run_dir = self.paths.training_run(mo_name)
        dirs = {k: run_dir / v for k, v in {"gt": "gt_wavs", "feat": "features", "f0": "f0", "f0c": "f0_coarse"}.items()}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        src = Path(ds_path) if ds_path else self.paths.dataset
        if not src.exists():
            src = self.paths.dataset
        audio_files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".wav", ".mp3", ".flac"))
        if not audio_files:
            raise FileNotFoundError(f"No audio files (.wav/.mp3/.flac) found in {src}")

        slicer = Slicer(tgt_sr)
        skipped = 0
        for idx0, audio_path in enumerate(audio_files):
            if stop_check and stop_check():
                raise TaskCancelled(f"Preprocessing cancelled after {idx0}/{len(audio_files)} file(s)")
            base = audio_path.stem
            say(idx0, len(audio_files), f"Preprocessing {base}...")
            self._purge_clips(run_dir, base)

            wav, orig_sr = load_audio(str(audio_path))
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            wav = self.pipeline.resample(wav.float().cpu(), orig_sr, tgt_sr)              # [1, T]
            wav = torch.from_numpy(zero_phase_highpass(wav.numpy(), tgt_sr, dsp.highpass_hz))

            max_val = wav.abs().max()
            if max_val > 0.005:                    # do not amplify silence
                wav = wav / max_val * (max_perc * alpha) + (1 - alpha) * wav

            for n, (s, e) in enumerate(slicer.get_indices(wav)):
                seg = wav[:, s:e]
                peak = seg.abs().max()
                if peak > 0.005:
                    seg = seg / peak * (max_perc * alpha) + (1 - alpha) * seg
                seg16 = self.pipeline.resample(seg, tgt_sr, FEAT_SR).clamp(-1, 1).to(self.device)

                units = self.pipeline.extract_features(seg16).squeeze(0).float()          # [T50, 768]
                n50 = units.shape[0]
                if n50 * 2 < min_frames:
                    skipped += 1
                    continue
                f0_raw, f0_coarse = self.pipeline.extract_f0(seg16, 0, n_frames=n50 * 2)
                gt = seg[:, : n50 * 2 * hop]

                name = f"{base}_{n}"
                save_audio(str(dirs["gt"] / f"{name}.wav"), gt.squeeze(0), tgt_sr)
                np.save(dirs["f0"] / f"{name}.wav.npy", f0_raw.cpu().numpy())
                np.save(dirs["f0c"] / f"{name}.wav.npy", f0_coarse.cpu().numpy())
                np.save(dirs["feat"] / f"{name}.npy", units.cpu().numpy())

            del wav
            _free_gpu()

        if skipped:
            log.warning("Skipped %d clip(s) shorter than %d frames", skipped, min_frames)
        say(1, 1, "Finished dataset preprocess!")

    @staticmethod
    def _purge_clips(run_dir: Path, base: str) -> None:
        """Remove clips (and cached spectrograms) left by an earlier preprocess of the same source
        file, so a different slicer/settings never leaves stale clips mixed into the dataset."""
        pat = re.compile(rf"^{re.escape(base)}_\d+\.(wav|wav\.npy|npy)$")
        for sub in [d for d in run_dir.iterdir() if d.is_dir() and (d.name in ("gt_wavs", "features", "f0", "f0_coarse")
                                                                    or d.name.startswith("specs_"))]:
            for f in sub.iterdir():
                if pat.match(f.name):
                    f.unlink()

    # ------------------------------------------------------------------ training
    @keep_awake
    def train_model(self, mo_name, epochs, frequency, batch, chart: Any, stop_check=None, progress_callback=None):
        say = lambda msg: progress_callback(0, 0, msg) if progress_callback else None
        mo_name = _require_model_name(mo_name)
        epochs, frequency, batch = int(epochs), int(frequency), int(batch)
        say("Started model training...")
        if chart is not None:
            chart.clear()

        say("Freeing up memory so training has as much headroom as possible...")
        with self._state_lock:
            self._training_active = True
            self.unload_model()
            if hasattr(self.pipeline, "unload_models"):
                self.pipeline.unload_models()
        _free_gpu()

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dev.type != "cuda":
            log.warning("No GPU found: training on the CPU will be marginally slower!")

        tcfg = self.cfg.train
        main_dir = self.paths.training_run(mo_name)
        wavs_dir = main_dir / "gt_wavs"
        wav_files = sorted(wavs_dir.glob("*.wav")) if wavs_dir.is_dir() else []
        if not wav_files:
            raise FileNotFoundError(f"No preprocessed audio in {wavs_dir}; run preprocessing first")

        profile: SRProfile = get_profile(audio_info(str(wav_files[0])).samplerate)
        say(f"Detected {profile.sr} Hz for {mo_name}")
        seed_everything(tcfg.seed)

        loader = val_ds = preview_ds = step_engine = opt_g = opt_d = net_d = net_g = batch_data = None
        completed = 0
        csv_file = None

        try:
            amp_dtype = resolve_amp_dtype(dev, tcfg.amp)

            net_g = SynthesizerTrnMs768NSFsid(**profile.model_kwargs(p_dropout=0.0, is_half=False)).to(dev)
            net_d = MultiPeriodDiscriminator().to(dev)

            g_params = sum(p.numel() for p in net_g.parameters())
            d_params = sum(p.numel() for p in net_d.parameters())
            total_params = g_params + d_params
            trainable_params = sum(p.numel() for p in net_g.parameters() if p.requires_grad) + sum(p.numel() for p in net_d.parameters() if p.requires_grad)

            log.info(
                "Training model parameters: G=%.2fM | D=%.2fM | total=%.2fM | trainable=%.2fM",
                g_params / 1e6, d_params / 1e6, total_params / 1e6, trainable_params / 1e6,
            )

            opt_g = AdamW(net_g.parameters(), lr=tcfg.lr, betas=tcfg.betas, eps=tcfg.eps)
            opt_d = AdamW(net_d.parameters(), lr=tcfg.lr, betas=tcfg.betas, eps=tcfg.eps)
            step_engine = TrainStep(net_g, net_d, opt_g, opt_d, profile, tcfg, dev)

            ckpts = sorted(
                ((int(m.group(1)), p) for p in main_dir.glob("ckpt_*.pth") if (m := re.fullmatch(r"ckpt_(\d+)\.pth", p.name))),
                reverse=True,
            )

            resuming = bool(ckpts)
            if ckpts:
                say(f"Resuming from checkpoint: {ckpts[0][1].name}...")
                data = load_checkpoint(ckpts[0][1], dev)
                load_state_logged(net_g, data["net_g"], "resume generator", raise_unexpected=True)
                load_state_logged(net_d, data["net_d"], "resume discriminator", raise_unexpected=True)
                opt_g.load_state_dict(data["opt_g"])
                opt_d.load_state_dict(data["opt_d"])
                completed = int(data.get("epoch", 0))
                step_engine._step_count = int(data.get("steps", 0)) * tcfg.grad_acc
            else:
                say("Loading pretrains...")
                for net, kind, label in ((net_g, "g", "generator"), (net_d, "d", "discriminator")):
                    candidates = [
                        self.paths.pretrained / getattr(profile, f"titan_{kind}"),
                        self.paths.pretrained / getattr(profile, f"pretrained_{kind}"),
                    ]
                    path = next((p for p in candidates if p.is_file()), None)
                    if path is None:
                        raise FileNotFoundError(
                            f"No pretrained {label} found for {profile.sr} Hz. Tried:\n"
                            + "\n".join(f"  {p}" for p in candidates)
                        )
                    state = load_checkpoint(path, dev)
                    state = {k.replace("module.", "", 1): v for k, v in state.get("model", state).items()}
                    load_state_logged(net, state, f"pretrained {label}", raise_unexpected=True)
                    say(f"Loaded {label}: {path}")

            if dev.type == "cpu":
                configure_cpu_runtime(tcfg)

            net_g, net_d = compile_cpu_training_models(net_g, net_d, tcfg, dev)

            train_names, val_names = split_names(main_dir, tcfg.val_fraction, tcfg.val_max, tcfg.seed)

            loader = create_dataloader(
                str(main_dir),
                profile,
                batch,
                tcfg.num_workers,
                names=train_names,
                cache=tcfg.cache_specs,
                seed=tcfg.seed,
                device=dev,
                amp_dtype=amp_dtype,
                check_vram=tcfg.check_vram,
                check_ram=tcfg.check_ram,
                vram_headroom_mb=tcfg.vram_headroom_mb,
                ram_headroom_mb=tcfg.ram_headroom_mb,
                model_params=total_params,
            )

            val_ds = RVCDataset(str(main_dir), profile, val_names, tcfg.cache_specs) if val_names else None
            say(f"{len(train_names)} training clips, {len(val_names)} held out for validation")

            backend = loader.dataset._cache_backend
            if backend == "gpu":
                say(f"Dataset cached in VRAM ({len(loader.dataset._gpu_cache)} clips)")
            elif backend == "ram":
                say(f"Dataset cached in RAM ({len(loader.dataset._ram_cache)} clips)")
            else:
                say("Dataset remains on disk!")

            preview_ds = val_ds or loader.dataset
            preview_dir = main_dir / "previews"
            preview_frames = int(tcfg.preview_seconds * 100)
            preview_dir.mkdir(exist_ok=True)

            ref = preview_dir / "reference.wav"
            if not ref.exists():
                gt = preview_ds[0][5][:preview_frames * profile.hop]
                save_audio(str(ref), gt, profile.sr)

            csv_path = main_dir / "train_log.csv"
            ema: dict = {}

            if resuming:
                if csv_path.exists():
                    with open(csv_path, "r", newline="", encoding="utf-8") as f:
                        rows = list(csv.reader(f))

                    header, body = (rows[0], rows[1:]) if rows else ([], [])

                    if completed > 0:
                        kept = [
                            r for r in body
                            if r
                            and r[0].lstrip("-").isdigit()
                            and int(r[0]) < completed
                        ]

                        if len(kept) != len(body):
                            log.info(
                                "Trimming %d stale row(s) from %s "
                                "(resuming at epoch %d)",
                                len(body) - len(kept),
                                csv_path.name,
                                completed,
                            )

                            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                                w = csv.writer(f)
                                if header:
                                    w.writerow(header)
                                w.writerows(kept)
                    else:
                        kept = body

                    if chart is not None:
                        a = tcfg.chart_ema
                        for row in kept:
                            if not row or len(row) < 8:
                                continue

                            ep = int(row[0])
                            mel_scaled = float(row[5]) * tcfg.c_mel

                            raw_vals = {
                                "g": float(row[3]),
                                "d": float(row[4]),
                                "mel": mel_scaled,
                                "fm": float(row[6]),
                                "kl": float(row[7]),
                            }

                            v_mel = 0.0
                            if len(row) > 8:
                                try:
                                    v = float(row[8])
                                    v_mel = 0.0 if math.isnan(v) else v
                                except ValueError:
                                    pass

                            for k, v in raw_vals.items():
                                ema[k] = (
                                    v
                                    if k not in ema or tcfg.chart_ema <= 0
                                    else tcfg.chart_ema * ema[k]
                                    + (1 - tcfg.chart_ema) * v
                                )

                            chart.update_plot(
                                ep,
                                ema["g"],
                                ema["d"],
                                ema["mel"],
                                ema["fm"],
                                ema["kl"],
                                v_mel * 20,
                            )

            else:
                # This is a genuinely new training run.
                kept = []

                if csv_path.exists():
                    csv_path.unlink()

            new_csv = not csv_path.exists()
            csv_file = open(csv_path, "a", newline="", encoding="utf-8")
            writer = csv.writer(csv_file)

            if new_csv:
                writer.writerow(["epoch", "opt_steps", "lr", "g_total", "d", "mel_raw", "fm", "kl", "val_mel"])

            steps_per_epoch = max(len(loader), 1)
            say(f"{steps_per_epoch} batches per epoch = {steps_per_epoch // tcfg.grad_acc} optimizer steps")

            for epoch in range(completed, epochs):
                lr = tcfg.lr * tcfg.lr_decay ** epoch

                for opt in (opt_g, opt_d):
                    for group in opt.param_groups:
                        group["lr"] = lr

                seed_everything(tcfg.seed + epoch)

                if hasattr(loader.batch_sampler, "set_epoch"):
                    loader.batch_sampler.set_epoch(epoch)

                acc = torch.zeros(len(_KEYS), device=dev)
                n_batches = 0

                for batch_data in loader:
                    if stop_check and stop_check():
                        say(f"Stopping after {completed} completed epoch(s)...")
                        self._save(main_dir, completed, net_g, net_d, opt_g, opt_d, f"ckpt_{completed}", steps=step_engine.opt_steps)
                        say("Finished model training!")
                        return

                    batch_data = [t.to(dev, non_blocking=True) for t in batch_data]
                    losses = step_engine(batch_data)
                    acc += torch.stack([losses[k] for k in _KEYS])
                    n_batches += 1

                avg = dict(zip(_KEYS, (acc / max(n_batches, 1)).tolist()))
                completed = epoch + 1
                val_mel = validate(step_engine, net_g, val_ds) if val_ds else float("nan")

                a = tcfg.chart_ema
                for k, v in avg.items():
                    ema[k] = v if k not in ema or a <= 0 else a * ema[k] + (1 - a) * v

                writer.writerow([
                    epoch,
                    step_engine.opt_steps,
                    f"{lr:.6e}",
                    avg["g"],
                    avg["d"],
                    avg["mel"] / tcfg.c_mel,
                    avg["fm"],
                    avg["kl"],
                    val_mel,
                ])
                csv_file.flush()

                msg = (
                    f"Epoch: {epoch:>3} | G: {avg['g']:7.3f} | D: {avg['d']:6.3f} | "
                    f"Mel: {avg['mel'] / tcfg.c_mel:6.3f} | FM: {avg['fm']:6.3f} | "
                    f"KL: {avg['kl']:6.4f} | Val: {val_mel:6.3f} | Learning Rate {lr:.2e}"
                )

                log.info(msg)

                if progress_callback: progress_callback(epoch, epochs, msg)
                chart.update_plot(epoch, ema["g"], ema["d"], ema["mel"], ema["fm"], ema["kl"], val_mel * 20)

                if frequency > 0 and completed % frequency == 0 and completed < epochs:
                    if progress_callback: progress_callback(epoch, epochs, f"Saving checkpoint at epoch {completed}...")
                    self._save_preview(completed, profile, step_engine, net_g, preview_dir, preview_ds, preview_frames)
                    self._save(main_dir, completed, net_g, net_d, opt_g, opt_d, f"ckpt_{completed}", steps=step_engine.opt_steps)

                    wdir = main_dir / "weights"
                    wdir.mkdir(exist_ok=True)
                    self._export_final(mo_name, net_g, profile, completed, dest=wdir / f"{mo_name}_e{completed}.pth")

            self._save_preview(epochs, profile, step_engine, net_g, preview_dir, preview_ds, preview_frames)
            self._save(main_dir, epochs, net_g, net_d, opt_g, opt_d, f"ckpt_{epochs}", steps=step_engine.opt_steps)
            final = self._export_final(mo_name, net_g, profile, epochs)
            say(f"Saved final model to: {final}")

        except Exception as exc:
            log.exception("Training failed")
            if _is_sticky_cuda_error(exc):
                log.error("CUDA context is unrecoverable (device-side assert). "
                         "Restart the application before training or converting again.")
            else:
                try:
                    if "step_engine" in locals():
                        self._save(main_dir, completed, net_g, net_d, opt_g, opt_d, "error_backup",
                                   steps=step_engine.opt_steps)   # not auto-resumed
                except Exception:
                    log.exception("Could not write error_backup")
            raise
        finally:
            self._training_active = False
            if csv_file:
                csv_file.close()
            loader = val_ds = preview_ds = step_engine = opt_g = opt_d = net_d = net_g = batch_data = None
            _free_gpu()

    def _save_preview(self, epochs, profile, step_engine, net_g, dir, dataset, frames):
        path = dir / f"epoch_{epochs:04d}.wav"
        try:
            pred, _, _ = synthesize(step_engine, net_g, dataset[0], frames)
            save_audio(str(path), pred, profile.sr)
        except Exception:
            log.exception("Could not write listening preview %s", path.name)
        finally:
            net_g.train()

    def _save(self, path: Path, epoch, net_g, net_d, opt_g, opt_d, name, steps: int = 0):
        """`epoch` = number of COMPLETED epochs (training resumes at that index). Written
        atomically so an interrupted save cannot leave a corrupt checkpoint."""
        target = Path(path) / f"{name}.pth"
        tmp = target.with_suffix(".tmp")
        torch.save({"epoch": epoch, "steps": steps, "net_g": net_g.state_dict(), "net_d": net_d.state_dict(),
                    "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()}, tmp)
        os.replace(tmp, target)

    def _export_final(self, name, net_g, profile: SRProfile, epoch, dest: Optional[Path] = None) -> Path:
        final_path = dest or (self.paths.models / f"{name}.pth")
        opt = OrderedDict()
        opt["weight"] = {k: v.half() for k, v in net_g.state_dict().items() if not k.startswith("enc_q")}
        opt["config"] = profile.checkpoint_config(p_dropout=0.0)
        opt["info"] = "%s - epoch %d" % (name, epoch)
        opt["sr"] = profile.sr
        opt["f0"] = 1
        opt["version"] = "v2"
        torch.save(opt, final_path)
        log.info("Model exported to %s", final_path)
        return final_path

    # ------------------------------------------------------------------ retrieval index
    @keep_awake
    def train_index(self, mo_name, progress_callback=None, stop_check=None):
        say = lambda i, msg: progress_callback(i, 4, msg) if progress_callback else None
        mo_name = _require_model_name(mo_name)
        say(0, "Started index training...")
        icfg = self.cfg.index

        feature_dir = self.paths.training_run(mo_name) / "features"
        files = sorted(feature_dir.glob("*.npy")) if feature_dir.is_dir() else []
        if not files:
            raise FileNotFoundError(f"The dataset for {mo_name} is not preprocessed yet")

        # ---- sample at most kmeans_train_max frames (memory-mapped, so nothing huge is loaded) ---
        say(1, "Loading features...")
        mm = [np.load(f, mmap_mode="r") for f in files]
        total = sum(m.shape[0] for m in mm)
        frac = min(1.0, icfg.kmeans_train_max / max(total, 1))
        rng = np.random.default_rng(icfg.seed)
        parts = []
        if frac < 1.0:
            for m in mm:
                take = max(1, int(round(m.shape[0] * frac)))
                parts.append(np.asarray(m[np.sort(rng.choice(m.shape[0], size=take, replace=False))], dtype=np.float32))
        else:
            parts = [np.asarray(m, dtype=np.float32) for m in mm]
        X = np.ascontiguousarray(np.concatenate(parts, axis=0))
        d = X.shape[1]
        log.info("Index: %d frames in dataset, %d used for clustering, dim %d", total, X.shape[0], d)

        if stop_check and stop_check():
            raise TaskCancelled("index training cancelled before clustering")

        # ---- reduce to cluster centres only if dataset is large (>200k frames / ~1.1h) ---
        # Upstream RVC logic: If frames <= 200,000, keep 100% raw features.
        # If clustering is required, cap centroids at (frames // 39) to eliminate FAISS warnings.
        cluster_threshold = getattr(icfg, "cluster_threshold", 200_000)

        if X.shape[0] > cluster_threshold and X.shape[0] > icfg.max_vectors:
            target_centroids = min(icfg.max_vectors, max(1, X.shape[0] // 39))
            say(2, f"Clustering {X.shape[0]} frames to {target_centroids} centroids...")
            km = faiss.Kmeans(d, target_centroids, niter=icfg.kmeans_iters, seed=icfg.seed, verbose=False)
            km.train(X)
            vectors = np.ascontiguousarray(km.centroids, dtype=np.float32)
        else:
            say(2, f"Using all {X.shape[0]} raw feature vectors (no clustering needed)...")
            vectors = X

        if stop_check and stop_check():
            raise TaskCancelled("index training cancelled before writing the index")
        say(3, "Building index...")

        n = vectors.shape[0]
        n_cells = min(max(int(round(math.sqrt(n))), 4), 4096)
        nprobe  = max(1, min(getattr(icfg, "nprobe", 1), n_cells))

        quantiser = faiss.IndexFlatL2(d)
        index     = faiss.IndexIVFFlat(quantiser, d, n_cells, faiss.METRIC_L2)
        index.train(vectors)

        batch_size = 8192
        for start in range(0, n, batch_size):
            index.add(vectors[start : start + batch_size])
        index.nprobe = nprobe

        # Filename encodes the IVF geometry so the user (and load_index_bank)
        # can see the cell count and nprobe at a glance — matching the original
        # naming convention:  added_IVF{n_cells}_Flat_nprobe_{nprobe}_{name}_v2.index
        index_name = f"added_IVF{n_cells}_Flat_nprobe_{nprobe}_{mo_name}_v2.index"
        index_path = self.paths.indices / index_name
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_path))
        say(4, f"Saved index ({n} vectors, IVF{n_cells}, nprobe={nprobe}) to: {index_path}")

    # ------------------------------------------------------------------ training sequence
    @keep_awake
    def sequence(self, mo_name, ds_path, tgt_sr, epochs, frequency, batch, chart,
                progress_callback=None, stop_check=None):
        """Run preprocess -> train -> index in sequence, skipping stages already complete."""
        say = lambda i, n, msg: progress_callback(i, n, msg) if progress_callback else None
        mo_name = _require_model_name(mo_name)
        run_dir = self.paths.training_run(mo_name)

        # ── Stage detection ───────────────────────────────────────────────────────
        def _preprocessed() -> bool:
            return all(
                (run_dir / sub).is_dir() and any((run_dir / sub).iterdir())
                for sub in ("gt_wavs", "features", "f0", "f0_coarse")
            )

        def _trained() -> bool:
            completed = max(
                (int(m.group(1)) for p in run_dir.glob("ckpt_*.pt*")
                if (m := re.search(r"ckpt_(\d+)", p.stem))),
                default=0
            )
            return completed >= epochs

        def _indexed() -> bool:
            return any(self.paths.indices.glob(f"*{mo_name}*.index"))

        # ── Stage 1: Preprocess ───────────────────────────────────────────────────
        if _preprocessed():
            say(0, 3, "Stage 1/3: Preprocessing already done, skipping...")
        else:
            say(0, 3, "Stage 1/3: Preprocessing dataset...")
            self.preprocess_dataset(mo_name, ds_path, tgt_sr,
                                    progress_callback=progress_callback,
                                    stop_check=stop_check)
            if stop_check and stop_check():
                return

        # ── Stage 2: Train ────────────────────────────────────────────────────────
        if _trained():
            say(1, 3, "Stage 2/3: Model already trained, skipping...")
        else:
            say(1, 3, "Stage 2/3: Training model...")
            self.train_model(mo_name, epochs, frequency, batch, chart,
                            progress_callback=progress_callback,
                            stop_check=stop_check)
            if stop_check and stop_check():
                return

        # ── Stage 3: Index ────────────────────────────────────────────────────────
        if _indexed():
            say(2, 3, "Stage 3/3: Index already built, skipping...")
        else:
            say(2, 3, "Stage 3/3: Building index...")
            self.train_index(mo_name,
                            progress_callback=progress_callback,
                            stop_check=stop_check)

        say(3, 3, "Sequence complete!")