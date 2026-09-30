"""
bin/pipeline.py - ContentVec + FCPE feature extraction, index retrieval, voice conversion,
                  and realtime engine.

Everything is organised around ONE primitive, `convert_window`, used by both paths:

    16 kHz audio, n frames of 10 ms  ->  model audio, exactly n * hop samples

Both paths then pass their audio through the same OutputChain (from lib.audio.dsp):
input high-pass -> output EQ -> loudness match -> soft clip. The chain is stateful so
block boundaries in realtime produce no clicks, and the offline path gets the same
treatment without any special-casing.

Realtime window layout (model domain, oldest -> newest, all sizes whole 10 ms frames):

    | extra: warm-up, discarded | crossfade + search: SOLA overlap | block: new audio |
"""
from __future__ import annotations

import gc
import math
import queue
import time
from typing import Callable, Optional

import faiss
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from torchfcpe import spawn_bundled_infer_model
from transformers import HubertModel

from bin.config import AppConfig, FEAT_HOP, FEAT_SR, FRAME_MS, RealtimeGeometry, TaskCancelled, \
    db_to_lin, pad_right_16k
from lib.audio.dsp import OutputChain, noise_gate, zero_phase_highpass
from bin.log import RealtimeStats, get_logger
from lib.audio.slicer import Slicer
from lib.ui.lslider import LiveParams

log = get_logger("pipeline")

ONSET_RAMP_MS = 2


# --------------------------------------------------------------------------- #
# Retrieval index
# --------------------------------------------------------------------------- #
class IndexBank:
    """Wraps a FAISS index for fast nearest-neighbour retrieval.

    Accepts either a pre-built faiss.Index (IVFFlat from train_index) or a raw
    numpy float32 array (legacy flat index — a fresh IVFFlat is built from the vectors).
    On CUDA, tries faiss-gpu and falls back to faiss-cpu silently.
    """

    def __init__(self, index_or_vectors, device: str):
        self.device = device

        if isinstance(index_or_vectors, faiss.Index):
            _index = index_or_vectors
            n, dim = _index.ntotal, _index.d
            vecs = np.empty((n, dim), dtype=np.float32)
            _index.reconstruct_n(0, n, vecs)
        else:
            vecs = np.ascontiguousarray(index_or_vectors, dtype=np.float32)
            n, dim = vecs.shape
            n_cells = min(max(int(round(math.sqrt(n))), 4), 4096)
            quantiser = faiss.IndexFlatL2(dim)
            _index = faiss.IndexIVFFlat(quantiser, dim, n_cells, faiss.METRIC_L2)
            _index.train(vecs)
            _index.add(vecs)
            _index.nprobe = max(1, min(32, n_cells))
            log.debug("IndexBank: built IVF%d from %d raw vectors", n_cells, n)

        self.n, self.dim = n, dim
        self._on_gpu = False
        if device.startswith("cuda"):
            try:
                gpu_id = int(device.split(":")[-1]) if ":" in device else 0
                res = faiss.StandardGpuResources()
                self._faiss = faiss.index_cpu_to_gpu(res, gpu_id, _index)
                self._on_gpu = True
                log.debug("IndexBank: faiss-gpu on %s (%d vectors)", device, n)
            except Exception:
                self._faiss = _index
                log.debug("IndexBank: faiss-gpu unavailable, using faiss-cpu (%d vectors)", n)
        else:
            self._faiss = _index
            log.debug("IndexBank: faiss-cpu (%d vectors)", n)

        self.vectors = torch.from_numpy(vecs).to(device)

    @property
    def nbytes(self) -> int:
        return self.vectors.numel() * self.vectors.element_size()

    def search(self, query: np.ndarray, k: int):
        """query: [Q, dim] float32 numpy. Returns (distances², indices) both [Q, k] numpy."""
        return self._faiss.search(query, k)


def load_index_bank(path, device, max_vectors: int, seed: int = 1234) -> IndexBank:
    """Read a faiss index file and prepare it for retrieval on `device`.

    IVFFlat files of the right size are used directly. Everything else (flat legacy
    indices, oversized IVFs) has its vectors reconstructed and a fresh IVF built.
    """
    index = faiss.read_index(str(path))
    total = int(index.ntotal)
    device_str = str(device)

    if isinstance(index, faiss.IndexIVFFlat) and total <= max_vectors:
        if total == 0:
            log.warning("Index %s is empty.", getattr(path, "name", path))
        return IndexBank(index, device_str)

    rng = np.random.default_rng(seed)
    keep_frac = min(1.0, max_vectors / max(total, 1))
    parts, step = [], 100_000
    for i0 in range(0, total, step):
        n = min(step, total - i0)
        chunk = index.reconstruct_n(i0, n)
        if keep_frac < 1.0:
            take = max(1, int(round(n * keep_frac)))
            chunk = chunk[np.sort(rng.choice(n, size=take, replace=False))]
        parts.append(np.ascontiguousarray(chunk, dtype=np.float32))
    vectors = np.concatenate(parts, axis=0) if parts else np.zeros((0, index.d), np.float32)
    del index
    if keep_frac < 1.0:
        log.warning("Index %s holds %d vectors; subsampled to %d. Re-run 'Train Index' to build a "
                    "compact index.", getattr(path, "name", path), total, vectors.shape[0])
    return IndexBank(vectors, device_str)


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #
class Pipeline:
    def __init__(self, cfg: AppConfig, device: str = "cuda", is_half: bool = False):
        self.cfg = cfg
        self.dsp = cfg.dsp
        self.device = device
        self.is_half = bool(is_half) and str(device).startswith("cuda")
        self.dtype = torch.float16 if self.is_half else torch.float32

        # ContentVec and pitch extractor are loaded on demand and released before training
        # (training reads precomputed features from disk and never calls these).
        self.pitch_extractor = None
        self.contentvec = None
        self._models_loaded = False
        self._load_models()

        self.f0_mel_min = 1127.0 * math.log(1.0 + self.dsp.f0_min / 700.0)
        self.f0_mel_max = 1127.0 * math.log(1.0 + self.dsp.f0_max / 700.0)

        self._sid = torch.zeros(1, dtype=torch.long, device=self.device)
        self._resamplers: dict = {}
        self._warned_len = False

    # ---- model lifecycle ---- #
    def _load_models(self) -> None:
        if self._models_loaded:
            return
        self.pitch_extractor = spawn_bundled_infer_model(self.device)
        contentvec = HubertModel.from_pretrained("lengyue233/content-vec-best")
        contentvec = contentvec.to(self.device).eval()
        self.contentvec = contentvec.half() if self.is_half else contentvec
        self._models_loaded = True

    def unload_models(self) -> None:
        """Free ContentVec + pitch extractor from VRAM before training."""
        self.contentvec = None
        self.pitch_extractor = None
        self._models_loaded = False
        gc.collect()
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- resampling ---- #
    def resampler(self, orig_sr: int, new_sr: int, device=None) -> T.Resample:
        device = str(device or self.device)
        key = (int(orig_sr), int(new_sr), device)
        r = self._resamplers.get(key)
        if r is None:
            try:
                r = T.Resample(int(orig_sr), int(new_sr), resampling_method="soxr_vhq").to(device)
            except Exception:
                r = T.Resample(int(orig_sr), int(new_sr),
                               lowpass_filter_width=64, rolloff=0.99).to(device)
            self._resamplers[key] = r
        return r

    def resample(self, x: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
        if orig_sr == new_sr or orig_sr <= 0 or new_sr <= 0:
            return x
        r = self.resampler(orig_sr, new_sr, x.device)
        if x.ndim == 1:
            return r(x.unsqueeze(0)).squeeze(0)
        elif x.ndim == 2:
            return r(x)
        else:
            shape = x.shape
            return r(x.reshape(-1, shape[-1])).reshape(*shape[:-1], -1)

    # ---- F0 ---- #
    @staticmethod
    def _median(f0: torch.Tensor, window: int) -> torch.Tensor:
        f0 = f0.reshape(-1)
        p = window // 2
        if window <= 1 or f0.numel() <= p:
            return f0
        x = F.pad(f0.view(1, 1, 1, -1), (p, p, 0, 0), mode="reflect")
        return x.unfold(3, window, 1).median(dim=-1)[0].reshape(-1)

    @staticmethod
    def _smooth(f0: torch.Tensor, window: int) -> torch.Tensor:
        """Hann smoothing over voiced frames only."""
        p = window // 2
        if window <= 1 or f0.numel() <= p:
            return f0
        v = (f0 > 0).to(f0.dtype).view(1, 1, -1)
        k = torch.hann_window(window + 2, periodic=False, device=f0.device, dtype=f0.dtype)[1:-1].view(1, 1, -1)
        num = F.conv1d(F.pad(f0.view(1, 1, -1) * v, (p, p), mode="reflect"), k)
        den = F.conv1d(F.pad(v, (p, p), mode="reflect"), k).clamp_min(1e-6)
        return torch.where(v.view(-1) > 0, (num / den).view(-1), torch.zeros_like(f0))

    @staticmethod
    def _fit(f0: torch.Tensor, n: int) -> torch.Tensor:
        m = f0.shape[0]
        if m == n:   return f0
        if m > n:    return f0[:n]
        if m == 0:   return f0.new_zeros(n)
        return torch.cat([f0, f0[-1:].expand(n - m)])

    def coarse_f0(self, f0: torch.Tensor) -> torch.Tensor:
        mel = 1127.0 * torch.log1p(f0 / 700.0)
        scaled = (mel - self.f0_mel_min) * (self.dsp.f0_bins - 2) / (self.f0_mel_max - self.f0_mel_min) + 1
        coarse = torch.where(mel > 0, scaled, torch.ones_like(scaled))
        return coarse.clamp(1, self.dsp.f0_bins - 1).round().long()

    def extract_f0(self, seg16: torch.Tensor, f0_up_key: float = 0, n_frames: Optional[int] = None):
        self._load_models()
        f0 = self.pitch_extractor.infer(
            seg16, sr=FEAT_SR, decoder_mode="local_argmax",
            f0_min=self.dsp.f0_min, f0_max=self.dsp.f0_max, interp_uv=False,
        )
        f0 = self._median(f0.reshape(-1), self.dsp.median_window)
        f0 = self._smooth(f0, self.dsp.smooth_window)
        if f0_up_key:
            f0 = f0 * (2.0 ** (f0_up_key / 12.0))
        if n_frames is not None:
            f0 = self._fit(f0, n_frames)
        return f0, self.coarse_f0(f0)

    def extract_features(self, seg16: torch.Tensor) -> torch.Tensor:
        """seg16: [1, T] at 16 kHz. Returns [1, T50, 768] in the model dtype."""
        self._load_models()
        with torch.inference_mode():
            out = self.contentvec(seg16.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            return out.hidden_states[12]

    def retrieve(self, feats: torch.Tensor, bank: IndexBank, rate: float) -> torch.Tensor:
        k = min(self.cfg.index.top_k, bank.n)
        q_np = feats.reshape(-1, feats.shape[-1]).float().cpu().numpy()
        D, I = bank.search(q_np, k)
        ix    = torch.from_numpy(I.astype(np.int64)).to(bank.vectors.device)
        score = torch.from_numpy(D.astype(np.float32)).to(bank.vectors.device)
        w = 1.0 / (score + 1e-12)
        w = w / w.sum(dim=1, keepdim=True)
        retrieved = (bank.vectors[ix] * w.unsqueeze(-1)).sum(dim=1).view_as(feats.float())
        return (retrieved * rate + feats.float() * (1.0 - rate)).to(feats.dtype)

    # ---- shared conversion primitive ---- #
    def convert_window(self, net_g, audio16: torch.Tensor, tgt_hop: int, *, f0_up_key: float = 0,
                       bank: Optional[IndexBank] = None, index_rate: float = 0.0,
                       protect: float = 0.5) -> torch.Tensor:
        """audio16: 1-D float32 on self.device, a whole number of 10 ms frames at 16 kHz.
        Returns a 1-D float32 of exactly n_frames * tgt_hop samples at the model rate."""
        n_frames, rem = divmod(audio16.shape[-1], FEAT_HOP)
        if rem or n_frames == 0:
            raise ValueError(f"window must be a whole number of 10 ms frames, got {audio16.shape[-1]} samples")

        with torch.inference_mode():
            x = audio16.view(1, -1)
            pad = pad_right_16k(n_frames)
            xp = F.pad(x.unsqueeze(0), (0, pad),
                       mode="reflect" if pad < x.shape[-1] else "replicate").squeeze(0) if pad else x

            feats0 = self.extract_features(xp)
            use_index = bank is not None and bank.n > 0 and index_rate > 0
            feats = self.retrieve(feats0, bank, index_rate) if use_index else feats0
            feats = feats.repeat_interleave(2, dim=1)[:, :n_frames]

            f0, f0_coarse = self.extract_f0(x, f0_up_key, n_frames)

            if use_index and protect < 0.5:
                orig = feats0.repeat_interleave(2, dim=1)[:, :n_frames]
                keep = (f0 > 0).to(torch.float32)
                keep = (keep + (1.0 - keep) * protect).view(1, -1, 1).to(feats.dtype)
                feats = feats * keep + orig * (1 - keep)

            p_len = torch.full((1,), n_frames, dtype=torch.long, device=self.device)
            out = net_g.infer(feats.contiguous(), p_len, f0_coarse.unsqueeze(0), f0.unsqueeze(0), self._sid)
            if isinstance(out, tuple):
                out = out[0]
            audio = out.reshape(-1).float()

        expected = n_frames * tgt_hop
        if audio.shape[0] != expected:
            if not self._warned_len:
                log.warning("model returned %d samples, expected %d (hop=%d, frames=%d)",
                            audio.shape[0], expected, tgt_hop, n_frames)
                self._warned_len = True
            audio = (audio[:expected] if audio.shape[0] > expected
                     else F.pad(audio, (0, expected - audio.shape[0])))
        return audio

    # ---- offline (whole file) ---- #
    def convert_file(self, net_g, tgt_sr: int, wav: torch.Tensor, orig_sr: int, *,
                     f0_up_key: float = 0, vol_scale: float = 1.0, protect: float = 0.33,
                     index_rate: float = 0.0, bank: Optional[IndexBank] = None,
                     progress_callback: Optional[Callable] = None,
                     stop_check: Optional[Callable[[], bool]] = None) -> torch.Tensor:
        """wav: mono audio at orig_sr. Returns a 1-D CPU tensor at tgt_sr, post-processed."""
        if net_g is None:
            raise ModuleNotFoundError("Generator not found! Cancelling inference!")
        dsp, hop = self.dsp, tgt_sr // 100
        chain = OutputChain(dsp, tgt_sr)

        if progress_callback: progress_callback(0, 6, "Resampling...")
        mono = torch.as_tensor(wav, dtype=torch.float32).reshape(-1).cpu()
        n_out = int(round(mono.numel() / orig_sr * tgt_sr))
        x16 = self.resample(mono, orig_sr, FEAT_SR)
        x16 = torch.from_numpy(zero_phase_highpass(x16.numpy(), FEAT_SR, dsp.highpass_hz))
        n_frames = max(1, -(-x16.numel() // FEAT_HOP))
        x16 = F.pad(x16, (0, n_frames * FEAT_HOP - x16.numel()))
        x16 = noise_gate(x16, FEAT_HOP, dsp.gate_db, dsp.gate_min_ms, dsp.gate_density).clamp(-1.0, 1.0)

        if progress_callback: progress_callback(0, 6, "Slicing segments...")
        chunks = Slicer(
            FEAT_SR, min_len_ms=10_000, target_len_ms=25_000,
            max_len_ms=35_000, min_silence_ms=400, min_keep_ms=0,
        ).get_indices(x16)
        ctx_frames = dsp.offline_context_ms // FRAME_MS
        out = torch.zeros(n_frames * hop, dtype=torch.float32)

        for i, (s, e) in enumerate(chunks):
            if stop_check and stop_check():
                raise TaskCancelled(f"Offline inference cancelled after {i+1}/{len(chunks)} chunks.")
            if progress_callback: progress_callback(i+1, len(chunks), f"Processing chunk {i+1}/{len(chunks)}...")
            sf, ef = int(s) // FEAT_HOP, min(n_frames, -(-int(e) // FEAT_HOP))
            if ef <= sf:
                continue
            c = min(ctx_frames, sf)
            seg = x16[(sf - c) * FEAT_HOP: ef * FEAT_HOP].to(self.device)
            audio = self.convert_window(net_g, seg, hop, f0_up_key=f0_up_key, bank=bank,
                                        index_rate=index_rate, protect=protect)
            out[sf * hop: ef * hop] = audio[c * hop:].cpu()
            del seg, audio

        out = out[:n_out] if out.numel() >= n_out else F.pad(out, (0, n_out - out.numel()))
        result = chain.apply_output(x16.numpy(), out.numpy(), vol_scale)

        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        return torch.from_numpy(np.ascontiguousarray(result))


# --------------------------------------------------------------------------- #
# Realtime
# --------------------------------------------------------------------------- #
class BlockConverter:
    """Pure conversion logic, no audio I/O.

    Feed consecutive mic blocks, get converted blocks back. Gating, ContentVec warm-up
    context, SOLA alignment, resampling, and the shared OutputChain are all handled here.
    """

    def __init__(self, cfg: AppConfig, pipeline: Pipeline, net_g, geo: RealtimeGeometry, *,
                 f0_up_key: float = 0, vol_scale: float = 1.0, protect: float = 0.33,
                 index_rate: float = 0.0, bank: Optional[IndexBank] = None, gate_db: float = -45.0):
        self.cfg, self.dsp, self.geo = cfg, cfg.dsp, geo
        self.pipe, self.net_g = pipeline, net_g
        self.dev = pipeline.device
        self.f0_up_key, self.vol_scale = f0_up_key, vol_scale
        self.protect, self.index_rate, self.bank = protect, index_rate, bank
        self.gate_db = gate_db
        self.gate_lin = db_to_lin(gate_db)
        self.hang_blocks = math.ceil(cfg.realtime.hangover_ms / (geo.block * FRAME_MS))

        # Shared output chain (input HP + output EQ + loudness match + soft clip)
        self.chain = OutputChain(cfg.dsp, geo.mic_sr)

        # SOLA crossfade constants
        cf = geo.crossfade_tgt
        t = torch.linspace(0.0, 1.0, steps=cf, device=self.dev)
        self.fade_in  = torch.sin(0.5 * math.pi * t) ** 2
        self.fade_out = 1.0 - self.fade_in
        self.ones_k   = torch.ones(1, 1, cf, device=self.dev)
        n_on = min(cf, max(1, int(geo.tgt_sr * ONSET_RAMP_MS / 1000)))
        self.onset = torch.ones(cf, device=self.dev)
        self.onset[:n_on] = torch.linspace(0.0, 1.0, steps=n_on, device=self.dev)

        self.reset()

    def reset(self) -> None:
        geo = self.geo
        self.ctx = np.zeros(geo.context_mic, dtype=np.float32)
        self.sola = torch.zeros(geo.crossfade_tgt, device=self.dev)
        self.sola_valid = False
        self.tail = torch.zeros(geo.tgt_zc, device=self.dev)
        self.hang = 0
        self.chain.reset()

    def _on_silence(self) -> None:
        self.sola_valid = False
        self.tail = torch.zeros_like(self.tail)
        self.chain.loud.reset()

    def warmup(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(2):
            self._convert(rng.normal(0, 0.01, self.geo.window_mic).astype(np.float32))
        self.reset()

    def is_active(self, x: np.ndarray) -> bool:
        z = self.geo.mic_zc
        n = (x.shape[0] // z) * z
        if n == 0:
            return False
        rms = np.sqrt(np.mean(np.square(x[:n].reshape(-1, z), dtype=np.float64), axis=1))
        return bool(rms.max() >= self.gate_lin)

    def process(self, block: np.ndarray):
        """block: float32 [block_mic]. Returns (out float32 [block_mic], state, infer_seconds)."""
        geo = self.geo
        if block.shape[0] != geo.block_mic:
            raise ValueError(f"expected {geo.block_mic} samples, got {block.shape[0]}")

        x = self.chain.apply_input_hp(block.astype(np.float32, copy=False))
        window = np.concatenate([self.ctx, x])
        self.ctx = window[-geo.context_mic:].copy()

        active = self.is_active(x)
        if active:
            self.hang = self.hang_blocks
        elif self.hang > 0:
            self.hang -= 1
            active = True

        if not active:
            self._on_silence()
            silence = np.zeros(geo.block_mic, dtype=np.float32)
            # Still run through eq so its state tracks real time (avoids a filter transient on re-open)
            return self.chain.apply_output(silence, silence, self.vol_scale), "idle", 0.0

        t0 = time.perf_counter()
        y = self._convert(window)
        return self.chain.apply_output(x, y, self.vol_scale), "converting", time.perf_counter() - t0

    def _convert(self, window: np.ndarray) -> np.ndarray:
        geo = self.geo
        with torch.inference_mode():
            w   = torch.from_numpy(window).to(self.dev)
            x16 = self.pipe.resample(w, geo.mic_sr, FEAT_SR)
            if x16.shape[0] != geo.window_16k:
                raise RuntimeError(f"resampled window has {x16.shape[0]} samples, expected {geo.window_16k}")
            x16 = noise_gate(x16, FEAT_HOP, self.gate_db, self.dsp.gate_min_ms,
                             self.dsp.gate_density).clamp(-1.0, 1.0)
            wide = self.pipe.convert_window(self.net_g, x16, geo.tgt_zc, f0_up_key=self.f0_up_key,
                                            bank=self.bank, index_rate=self.index_rate,
                                            protect=self.protect)
            play, look = self._sola(wide)

            if geo.tgt_sr != geo.mic_sr:
                seg = torch.cat([self.tail, play, look])
                seg = self.pipe.resample(seg, geo.tgt_sr, geo.mic_sr)
                out = seg[geo.mic_zc: geo.mic_zc + geo.block_mic]
            else:
                out = play
            self.tail = play[-geo.tgt_zc:].clone()
            return out.cpu().numpy()

    def _sola(self, wide: torch.Tensor):
        g = self.geo
        e, cf, se, blk = g.extra_tgt, g.crossfade_tgt, g.search_tgt, g.block_tgt
        overlap = wide[e: e + cf + se]

        if self.sola_valid:
            x, k = overlap.view(1, 1, -1), self.sola.view(1, 1, -1)
            num  = F.conv1d(x, k)
            den  = torch.sqrt(F.conv1d(x * x, self.ones_k) + 1e-8)
            best = int(torch.argmax((num / den).view(-1)))
            blended = self.sola * self.fade_out + wide[e + best: e + best + cf] * self.fade_in
        else:
            best    = 0
            blended = wide[e: e + cf] * self.onset

        valid = torch.cat([blended, wide[e + best + cf:]])
        self.sola = valid[blk: blk + cf].clone()
        self.sola_valid = True
        return valid[:blk], valid[blk: blk + g.tgt_zc]


# --------------------------------------------------------------------------- #
# Device I/O
# --------------------------------------------------------------------------- #
def pick_stream_sr(sd, in_idx: int, out_idx: int, preferred: int) -> int:
    dev_in, dev_out = sd.query_devices(in_idx), sd.query_devices(out_idx)
    cands = []
    for c in (preferred, int(dev_in["default_samplerate"]), int(dev_out["default_samplerate"]), 48000, 44100):
        if c and c % 100 == 0 and c not in cands:
            cands.append(c)
    for c in cands:
        try:
            sd.check_input_settings(device=in_idx, samplerate=c, channels=1, dtype="float32")
            sd.check_output_settings(device=out_idx, samplerate=c, channels=1, dtype="float32")
            return c
        except Exception as exc:
            log.debug("stream rate %d rejected: %s", c, exc)
    raise RuntimeError(f"No common sample rate for devices {in_idx}/{out_idx} (tried {cands})")


class RealtimeEngine:
    def __init__(self, cfg: AppConfig, pipeline: Pipeline):
        self.cfg, self.pipeline = cfg, pipeline

    def run(self, net_g, tgt_sr: int, bank: Optional[IndexBank], mic_idx: int, spk_idx: int, *,
            live_params: LiveParams, block_ms: int, stop_check: Optional[Callable[[], bool]] = None,
            progress_callback: Optional[Callable] = None) -> None:
        import sounddevice as sd

        rt = self.cfg.realtime
        ic = self.cfg.inference
        sr  = pick_stream_sr(sd, mic_idx, spk_idx, rt.stream_sr)
        geo = RealtimeGeometry.build(rt, sr, tgt_sr, block_ms)

        conv = BlockConverter(
            self.cfg, self.pipeline, net_g, geo,
            f0_up_key=live_params.get("f0_up_key", ic.f0_up_key),
            vol_scale=live_params.get("vol_scale", ic.vol_scale),
            protect=live_params.get("protect", ic.protect),
            index_rate=live_params.get("index_rate", ic.idx_rate),
            bank=bank,
            gate_db=live_params.get("gate_db", rt.gate_db),
        )
        stats   = RealtimeStats(log, rt.report_every_s)
        block_s = geo.block * FRAME_MS / 1000.0

        log.info("Realtime start: model %d Hz, stream %d Hz, block %d ms, context %d ms, "
                 "algorithmic latency %d ms, gate %.0f dB, hangover %d block(s)",
                 tgt_sr, sr, geo.block * FRAME_MS, geo.context_frames * FRAME_MS,
                 geo.algorithmic_latency_ms, live_params.get("gate_db", rt.gate_db), conv.hang_blocks)
        conv.warmup()

        in_q: queue.SimpleQueue  = queue.SimpleQueue()
        out_q: queue.SimpleQueue = queue.SimpleQueue()
        cb = {"cur": None, "off": 0, "underruns": 0, "flag_events": 0, "primed": False}

        def callback(indata, outdata, frames, time_info, status):
            if status:
                cb["flag_events"] += 1
            in_q.put(indata[:, 0].copy())
            out, pos = outdata[:, 0], 0
            while pos < frames:
                cur = cb["cur"]
                if cur is None:
                    try:
                        cur = out_q.get_nowait()
                    except queue.Empty:
                        break
                    cb["cur"], cb["off"] = cur, 0
                n = min(frames - pos, len(cur) - cb["off"])
                out[pos: pos + n] = cur[cb["off"]: cb["off"] + n]
                pos += n;  cb["off"] += n
                if cb["off"] >= len(cur):
                    cb["cur"] = None
            if pos < frames:
                out[pos:] = 0.0
                if cb["primed"]:
                    cb["underruns"] += 1

        def status_msg(state: str) -> str:
            return f"Audio stream status: Model - {tgt_sr} Hz | Output - {sr} Hz | State - {state}"

        dev_block   = max(1, sr * rt.device_block_ms // 1000)
        max_backlog = rt.max_backlog_blocks * geo.block_mic
        parts, n_have, last_state, last_under = [], 0, None, 0

        with sd.Stream(device=(mic_idx, spk_idx), samplerate=sr, blocksize=dev_block, dtype="float32",
                       channels=1, latency=rt.latency, callback=callback) as stream:
            log.info("Stream open: latency in/out = %s s", stream.latency)
            try:
                while not (stop_check and stop_check()):
                    try:
                        chunk = in_q.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    parts.append(chunk);  n_have += len(chunk)
                    while True:
                        try:
                            chunk = in_q.get_nowait()
                        except queue.Empty:
                            break
                        parts.append(chunk);  n_have += len(chunk)

                    if n_have < geo.block_mic:
                        continue
                    data = np.concatenate(parts)
                    if len(data) - geo.block_mic > max_backlog:
                        drop = (len(data) // geo.block_mic - rt.max_backlog_blocks) * geo.block_mic
                        data = data[drop:]
                        conv.reset();  stats.dropped()
                        log.warning("Realtime fell behind; dropped %.0f ms of input", drop * 1000 / sr)

                    while len(data) >= geo.block_mic:
                        conv.f0_up_key  = live_params.get("f0_up_key",  conv.f0_up_key)
                        conv.vol_scale  = live_params.get("vol_scale",  conv.vol_scale)
                        conv.protect    = live_params.get("protect",    conv.protect)
                        conv.index_rate = live_params.get("index_rate", conv.index_rate)
                        conv.gate_db    = live_params.get("gate_db",    conv.gate_db)

                        blk, data = data[: geo.block_mic], data[geo.block_mic:]
                        out, state, infer_s = conv.process(blk)
                        stats.record(infer_s, block_s) if state == "converting" else stats.skipped()
                        if not cb["primed"]:
                            out_q.put(np.zeros(sr * rt.prebuffer_ms // 1000, dtype=np.float32))
                            cb["primed"] = True
                        out_q.put(out)
                        if progress_callback and state != last_state:
                            progress_callback(0, 0, status_msg("Inference" if state == "converting" else "Idle"))
                            last_state = state
                    parts, n_have = [data], len(data)

                    for _ in range(cb["underruns"] - last_under):
                        stats.underrun()
                    last_under = cb["underruns"]
                    stats.report()
            finally:
                stats.report(force=True)
                if cb["flag_events"]:
                    log.info("PortAudio reported %d over/underflow status events", cb["flag_events"])
        log.info("Realtime stopped")