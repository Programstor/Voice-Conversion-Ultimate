"""
bin/realtime.py - realtime voice conversion.

  BlockConverter  : pure logic, no audio I/O. Feed it consecutive mic blocks, it returns the
                    converted block (same length) - gating, ContentVec warm-up context, SOLA
                    alignment, resampling, EQ and loudness matching. Testable without a device.
  RealtimeEngine  : wraps BlockConverter in a sounddevice *callback* stream. The PortAudio
                    thread only copies samples in/out of queues; inference runs in the calling
                    (task) thread, so a slow block can no longer block audio capture.

Window layout for one step (model domain, oldest -> newest), all sizes whole 10 ms frames:

    | extra: warm-up, discarded | crossfade + search: SOLA overlap | block: new audio |

Every step outputs exactly one block, so the stream never drifts and nothing is time-stretched.
"""
from __future__ import annotations

import math
import queue
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from bin.config import AppConfig, FEAT_HOP, FEAT_SR, FRAME_MS, RealtimeGeometry, db_to_lin
from lib.audio.dsp import LoudnessMatcher, StatefulSOS, highpass_sos, noise_gate, output_sos
from bin.log import RealtimeStats, get_logger
from bin.pipeline import IndexBank, Pipeline

from lib.ui.lslider import LiveParams

log = get_logger("realtime")

ONSET_RAMP_MS = 2   # fade-in used for the first block after silence (no previous audio to blend with)


class BlockConverter:
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

        block_ms = geo.block * FRAME_MS
        self.hang_blocks = math.ceil(cfg.realtime.hangover_ms / block_ms)

        # stateful filters: input high-pass on the NEW block only, output EQ chain on the played audio
        self.hp = StatefulSOS(highpass_sos(geo.mic_sr, self.dsp.highpass_hz))
        self.out_chain = StatefulSOS(output_sos(self.dsp, geo.mic_sr))
        self.loud = LoudnessMatcher(self.dsp.rms_smoothing, self.dsp.max_rms_gain)

        # SOLA constants (constant-gain sine-squared crossfade, energy-normalised correlation)
        cf = geo.crossfade_tgt
        t = torch.linspace(0.0, 1.0, steps=cf, device=self.dev)
        self.fade_in = torch.sin(0.5 * math.pi * t) ** 2
        self.fade_out = 1.0 - self.fade_in
        self.ones_k = torch.ones(1, 1, cf, device=self.dev)
        n_on = min(cf, max(1, int(geo.tgt_sr * ONSET_RAMP_MS / 1000)))
        self.onset = torch.ones(cf, device=self.dev)
        self.onset[:n_on] = torch.linspace(0.0, 1.0, steps=n_on, device=self.dev)

        self.reset()

    # ---- state ---- #
    def reset(self) -> None:
        geo = self.geo
        self.ctx = np.zeros(geo.context_mic, dtype=np.float32)   # filtered mic history
        self.sola = torch.zeros(geo.crossfade_tgt, device=self.dev)
        self.sola_valid = False
        self.tail = torch.zeros(geo.tgt_zc, device=self.dev)     # last 10 ms played (resampler margin)
        self.hang = 0
        self.loud.reset()

    def _on_silence(self) -> None:
        self.sola_valid = False
        self.tail = torch.zeros_like(self.tail)
        self.loud.reset()

    def warmup(self) -> None:
        """Run one conversion on low-level noise so kernels/allocations are ready before audio starts."""
        rng = np.random.default_rng(0)
        for _ in range(2):
            self._convert(rng.normal(0, 0.01, self.geo.window_mic).astype(np.float32))
        self.reset()

    # ------------------------------------------------------------------ one step
    def is_active(self, x: np.ndarray) -> bool:
        """True when any 10 ms frame reaches the gate level (numpy: no GPU sync)."""
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

        x = self.hp(block.astype(np.float32, copy=False))
        window = np.concatenate([self.ctx, x])
        self.ctx = window[-geo.context_mic:].copy()

        active = self.is_active(x)
        if active:
            self.hang = self.hang_blocks
        elif self.hang > 0:                 # hangover: keep converting so word endings are not cut
            self.hang -= 1
            active = True

        if not active:
            self._on_silence()
            return self.out_chain(np.zeros(geo.block_mic, dtype=np.float32)), "idle", 0.0

        t0 = time.perf_counter()
        y = self._convert(window)
        y = self.out_chain(y)
        y = self.loud(x, y, self.vol_scale)
        return np.tanh(y).astype(np.float32, copy=False), "converting", time.perf_counter() - t0

    def _convert(self, window: np.ndarray) -> np.ndarray:
        geo = self.geo
        with torch.inference_mode():
            w = torch.from_numpy(window).to(self.dev)
            x16 = self.pipe.resample(w, geo.mic_sr, FEAT_SR)
            if x16.shape[0] != geo.window_16k:
                raise RuntimeError(f"resampled window has {x16.shape[0]} samples, expected {geo.window_16k}")
            x16 = noise_gate(x16, FEAT_HOP, self.gate_db, self.dsp.gate_min_ms,
                             self.dsp.gate_density).clamp(-1.0, 1.0)
            wide = self.pipe.convert_window(self.net_g, x16, geo.tgt_zc, f0_up_key=self.f0_up_key,
                                            bank=self.bank, index_rate=self.index_rate,
                                            protect=self.protect)
            play, look = self._sola(wide)

            # Resample with one frame of margin on each side so block joins do not get the
            # resampler's edge roll-off (which would put a small dip/click at every block boundary).
            if geo.tgt_sr != geo.mic_sr:
                seg = torch.cat([self.tail, play, look])
                seg = self.pipe.resample(seg, geo.tgt_sr, geo.mic_sr)
                out = seg[geo.mic_zc: geo.mic_zc + geo.block_mic]
            else:
                out = play
            self.tail = play[-geo.tgt_zc:].clone()
            return out.cpu().numpy()          # the only host sync of the step

    def _sola(self, wide: torch.Tensor):
        """Align the new window with the previous block's tail (energy-normalised correlation),
        crossfade, and cut exactly one block. Returns (block, one-frame look-ahead)."""
        g = self.geo
        e, cf, se, blk = g.extra_tgt, g.crossfade_tgt, g.search_tgt, g.block_tgt
        overlap = wide[e: e + cf + se]

        if self.sola_valid:
            x, k = overlap.view(1, 1, -1), self.sola.view(1, 1, -1)
            num = F.conv1d(x, k)
            den = torch.sqrt(F.conv1d(x * x, self.ones_k) + 1e-8)
            best = int(torch.argmax((num / den).view(-1)))
            blended = self.sola * self.fade_out + wide[e + best: e + best + cf] * self.fade_in
        else:
            best = 0
            blended = wide[e: e + cf] * self.onset

        valid = torch.cat([blended, wide[e + best + cf:]])     # length cf + se + blk - best >= blk + cf
        self.sola = valid[blk: blk + cf].clone()
        self.sola_valid = True
        return valid[:blk], valid[blk: blk + g.tgt_zc]


# --------------------------------------------------------------------------- #
# Device I/O
# --------------------------------------------------------------------------- #
def pick_stream_sr(sd, in_idx: int, out_idx: int, preferred: int) -> int:
    """First sample rate (multiple of 100 Hz) that both devices accept."""
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
        except Exception as exc:  # PortAudioError, ValueError
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
        sr = pick_stream_sr(sd, mic_idx, spk_idx, rt.stream_sr)
        geo = RealtimeGeometry.build(rt, sr, tgt_sr, block_ms)

        conv = BlockConverter(
            self.cfg, self.pipeline, net_g, geo, 
            f0_up_key=live_params.get("f0_up_key", ic.f0_up_key),
            vol_scale=live_params.get("vol_scale", ic.vol_scale), 
            protect=live_params.get("protect", ic.protect), 
            index_rate=live_params.get("index_rate", ic.idx_rate),
            bank=bank, 
            gate_db=live_params.get("gate_db", rt.gate_db)
        )
        stats = RealtimeStats(log, rt.report_every_s)
        block_s = geo.block * FRAME_MS / 1000.0

        log.info("Realtime start: model %d Hz, stream %d Hz, block %d ms, context %d ms, "
                 "algorithmic latency %d ms, gate %.0f dB, hangover %d block(s)",
                 tgt_sr, sr, geo.block * FRAME_MS, geo.context_frames * FRAME_MS,
                 geo.algorithmic_latency_ms, live_params.get("gate_db", rt.gate_db), conv.hang_blocks)
        conv.warmup()

        in_q: queue.SimpleQueue = queue.SimpleQueue()
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
                pos += n
                cb["off"] += n
                if cb["off"] >= len(cur):
                    cb["cur"] = None
            if pos < frames:
                out[pos:] = 0.0
                if cb["primed"]:
                    cb["underruns"] += 1

        def status_msg(state: str) -> str:
            return f"Audio stream status: Model - {tgt_sr} Hz | Output - {sr} Hz | State - {state}"

        dev_block = max(1, sr * rt.device_block_ms // 1000)
        parts, n_have, last_state, last_under = [], 0, None, 0
        max_backlog = rt.max_backlog_blocks * geo.block_mic

        with sd.Stream(device=(mic_idx, spk_idx), samplerate=sr, blocksize=dev_block, dtype="float32",
                       channels=1, latency=rt.latency, callback=callback) as stream:
            log.info("Stream open: latency in/out = %s s", stream.latency)
            try:
                while not (stop_check and stop_check()):
                    try:
                        chunk = in_q.get(timeout=0.05)
                    except queue.Empty:
                        continue
                    parts.append(chunk)
                    n_have += len(chunk)
                    while True:                       # drain what else is already waiting
                        try:
                            chunk = in_q.get_nowait()
                        except queue.Empty:
                            break
                        parts.append(chunk)
                        n_have += len(chunk)

                    if n_have < geo.block_mic:
                        continue
                    data = np.concatenate(parts)
                    if len(data) - geo.block_mic > max_backlog:     # far behind: drop oldest, resync
                        drop = (len(data) // geo.block_mic - rt.max_backlog_blocks) * geo.block_mic
                        data = data[drop:]
                        conv.reset()
                        stats.dropped()
                        log.warning("Realtime fell behind; dropped %.0f ms of input", drop * 1000 / sr)

                    while len(data) >= geo.block_mic:
                        conv.f0_up_key = live_params.get("f0_up_key", conv.f0_up_key)
                        conv.vol_scale = live_params.get("vol_scale", conv.vol_scale)
                        conv.protect = live_params.get("protect", conv.protect)
                        conv.index_rate = live_params.get("index_rate", conv.index_rate)
                        conv.gate_db = live_params.get("gate_db", conv.gate_db)
                        
                        blk, data = data[: geo.block_mic], data[geo.block_mic:]
                        out, state, infer_s = conv.process(blk)
                        if state == "converting":
                            stats.record(infer_s, block_s)
                        else:
                            stats.skipped()
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
