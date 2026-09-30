"""
audio_io.py - load and save audio with soundfile, returning/accepting torch tensors.

Why: recent torchaudio releases moved torchaudio.load / torchaudio.save to TorchCodec, which is
a separate package and needs FFmpeg shared libraries on the system ("TorchCodec is required for
load_with_torchcodec"). soundfile is already a dependency, ships its own libsndfile, and reads
WAV / FLAC / OGG / Opus / MP3 without FFmpeg.

Suggested location: lib/audio/audio_io.py

    from lib.audio.audio_io import load_audio, save_audio
    wav, sr = load_audio(path)                    # float32 tensor [channels, frames], sample rate
    save_audio("out.mp3", wav, 44100)             # wav: [channels, frames] or [frames]
"""
from __future__ import annotations

import logging
import os

import numpy as np
import soundfile as sf
import torch

log = logging.getLogger("rvc.audio_io")

# libsndfile's MP3 writer (LAME) only accepts the standard MPEG sample rates.
MP3_RATES = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)


class AudioFormatError(RuntimeError):
    """The file exists but libsndfile cannot decode it."""


def _decode(path):
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)      # [frames, channels]
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise AudioFormatError(
            f"Cannot read '{os.path.basename(path)}': {exc}. Supported: WAV, FLAC, OGG/Opus, MP3. "
            f"Convert other formats (m4a, aac, wma, ...) first, e.g. with ffmpeg."
        ) from exc
    return data, int(sr)


def load_audio(path):
    """Same contract as torchaudio.load: (float32 tensor [channels, frames], sample_rate)."""
    data, sr = _decode(path)
    return torch.from_numpy(np.ascontiguousarray(data.T)), sr


def load_audio_playback(path):
    """(float32 numpy array [frames, channels], sample_rate) - the shape sounddevice/AudioPlayer want."""
    data, sr = _decode(path)
    return np.ascontiguousarray(data), sr


def save_audio(path, wav, sample_rate, fmt=None, subtype=None):
    """Like torchaudio.save. The format comes from the file extension unless fmt is given."""
    path = os.fspath(path)
    sample_rate = int(sample_rate)
    wav = wav.detach().cpu().float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    if path.lower().endswith(".mp3") and sample_rate not in MP3_RATES:
        target = min(MP3_RATES, key=lambda r: abs(r - sample_rate))
        log.warning("MP3 does not support %d Hz; resampling to %d Hz before saving", sample_rate, target)
        import torchaudio.functional as AF
        wav = AF.resample(wav, sample_rate, target)
        sample_rate = target

    arr = wav.numpy().T  # [samples, channels]
    peak = np.abs(arr).max()
    if peak > 0.99:
        arr = arr / peak
    sf.write(path, np.ascontiguousarray(arr), sample_rate, format=fmt, subtype=subtype)

def audio_info(path):
    """Cheap metadata read (samplerate, frames, ...) without decoding samples."""
    return sf.info(os.fspath(path))