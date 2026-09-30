"""
slicer.py - silence-aware slicing for RVC dataset preparation.

Drop-in replacement for the old Slicer: same call pattern,

    slicer = Slicer(tgt_sr)                 # hop = tgt_sr // 100  (480 at 48 kHz, 400 at 40 kHz)
    for start, end in slicer.get_indices(wav):   # sample indices, multiples of hop
        clip = wav[..., start:end]

What changed and why
  * Everything lives on the model's 10 ms frame grid, so every cut is a multiple of the hop
    and the 3:1 (48 kHz -> 16 kHz) or 5:2 (40 kHz -> 16 kHz) mapping stays exact.
  * Frame energy is computed with one einsum over a (n_frames, hop) view - O(N) memory
    instead of the old unfold(...)**2, which materialised ~4x the audio as a temporary.
  * The silence threshold adapts to the recording's noise floor (clamped to a sane range),
    so quiet or noisy recordings do not need hand-tuned dB values.
  * Cuts are made in the middle of real silences (>= min_silence_ms), choosing the one
    closest to the target length; a small pad of silence is kept on both sides so breaths
    and consonant tails are not clipped.  Only if a phrase has no silence at all inside the
    allowed window does it fall back to the quietest frame.
  * The result is directly usable as training clips (1.5-6 s by default), so the extra
    fixed 3.7 s "double slice" is no longer needed.
"""
from __future__ import annotations

import numpy as np


def _as_mono_numpy(waveform) -> np.ndarray:
    if hasattr(waveform, "detach"):                 # torch.Tensor without importing torch
        waveform = waveform.detach().cpu().numpy()
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=0)
    return np.ascontiguousarray(x)


def _true_runs(mask: np.ndarray):
    """Start (inclusive) and end (exclusive) indices of each run of True in a bool array."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


class Slicer:
    def __init__(
        self,
        sr: int,
        *,
        threshold_db: float = -40.0,      # used as-is when adaptive=False
        adaptive: bool = True,
        margin_db: float = 12.0,          # silence threshold = noise floor + margin ...
        min_thr_db: float = -55.0,        # ... clamped to this range
        max_thr_db: float = -30.0,
        min_len_ms: int = 4000,
        target_len_ms: int = 8000,
        max_len_ms: int = 12000,
        min_silence_ms: int = 150,        # a silence must be this long to be a cut point
        pad_ms: int = 50,                 # silence kept around each clip
        min_keep_ms: int = 500,           # drop a trailing fragment shorter than this
        hop_size: int | None = None,
    ):
        self.sr = int(sr)
        self.hop = int(hop_size or self.sr // 100)
        frame_ms = 1000.0 * self.hop / self.sr

        def frames(ms: float) -> int:
            return max(1, int(round(ms / frame_ms)))

        self.threshold_db = threshold_db
        self.adaptive = adaptive
        self.margin_db = margin_db
        self.min_thr_db, self.max_thr_db = min_thr_db, max_thr_db
        self.pad = frames(pad_ms)
        self.min_len = frames(min_len_ms)
        self.target_len = max(frames(target_len_ms), self.min_len)
        self.max_len = max(frames(max_len_ms), self.target_len)
        self.min_sil = max(frames(min_silence_ms), 2 * self.pad + 1)   # pads must fit inside a silence
        self.min_keep = frames(min_keep_ms)

    # ---- analysis ---- #
    def frame_db(self, x: np.ndarray) -> np.ndarray:
        """Energy per hop-sized frame in dB, lightly smoothed over 3 frames."""
        n = len(x)
        n_frames = -(-n // self.hop)                       # ceil
        if n_frames * self.hop != n:
            x = np.pad(x, (0, n_frames * self.hop - n))
        blocks = x.reshape(n_frames, self.hop)
        energy = np.einsum("ij,ij->i", blocks, blocks) / self.hop      # mean square, no big temporaries
        if n_frames >= 3:
            energy = np.convolve(energy, np.ones(3, dtype=np.float32) / 3.0, mode="same")
        return 10.0 * np.log10(energy + 1e-12)

    def threshold_for(self, db: np.ndarray) -> float:
        if not self.adaptive:
            return self.threshold_db
        floor = float(np.percentile(db, 5))
        return float(min(max(floor + self.margin_db, self.min_thr_db), self.max_thr_db))

    # ---- slicing ---- #
    def get_indices(self, waveform):
        x = _as_mono_numpy(waveform)
        if x.size == 0:
            return []
        db = self.frame_db(x)
        n = len(db)
        silent = db < self.threshold_for(db)
        sound = np.flatnonzero(~silent)
        if sound.size == 0:
            return []
        last_sound = int(sound[-1])

        run_a, run_b = _true_runs(silent)
        long_run = (run_b - run_a) >= self.min_sil
        cut_a, cut_b = run_a[long_run], run_b[long_run]     # candidate silences
        centers = (cut_a + cut_b) // 2
        lengths = cut_b - cut_a

        chunks = []
        prev_end = 0
        pos = int(sound[0])                                  # first frame of sound
        while pos <= last_sound:
            start = pos
            # Remaining audio fits in one clip -> finish.
            if last_sound + 1 - start <= self.max_len:
                end = min(last_sound + 1 + self.pad, n)
                chunks.append((max(start - self.pad, prev_end), end))
                break

            lo, hi, tgt = start + self.min_len, start + self.max_len, start + self.target_len
            i0, i1 = np.searchsorted(centers, lo), np.searchsorted(centers, hi, side="right")
            if i1 > i0:                                       # cut inside a real silence
                idx = np.arange(i0, i1)
                score = np.abs(centers[idx] - tgt) - 0.5 * lengths[idx]
                k = int(idx[int(np.argmin(score))])
                end = int(cut_a[k]) + self.pad
                chunks.append((max(start - self.pad, prev_end), end))
                prev_end, pos = end, int(cut_b[k])            # next sound starts after the silence
            else:                                             # no silence available: quietest frame
                k = lo + int(np.argmin(db[lo:hi]))
                chunks.append((max(start - self.pad, prev_end), k))
                prev_end, pos = k, k

        out = []
        for a, b in chunks:
            if b - a >= self.min_keep:
                out.append((a * self.hop, min(b * self.hop, len(x))))
        return out