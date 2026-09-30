from __future__ import annotations

import contextlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Optional

ROOT_LOGGER = "rvc"
FMT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-14s | %(threadName)-18s | %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False
_log_path: Optional[Path] = None


def get_logger(name: str = "") -> logging.Logger:
    """get_logger('realtime') -> 'rvc.realtime'. Pass __name__-style short names."""
    return logging.getLogger(f"{ROOT_LOGGER}.{name}" if name else ROOT_LOGGER)


# --------------------------------------------------------------------------- #
# System information
# --------------------------------------------------------------------------- #
_PACKAGES = (
    "torch", "torchaudio", "torchfcpe", "transformers", "numpy", "scipy", "librosa"
    "soundfile", "sounddevice", "faiss-cpu", "faiss-gpu", "PyQt6", "pymss", "onnxruntime",
)
_ENV_VARS = (
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION",
    "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "QT_QPA_PLATFORM", "VCU_HOME",
)


def _pkg_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _ram_gb() -> Optional[str]:
    try:
        import psutil  # optional
        vm = psutil.virtual_memory()
        return f"{vm.total / 2**30:.1f} GB total, {vm.available / 2**30:.1f} GB free"
    except Exception:
        pass
    try:  # Linux fallback
        info = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                info[key] = int(rest.split()[0])
        return f"{info['MemTotal'] / 2**20:.1f} GB total, {info.get('MemAvailable', 0) / 2**20:.1f} GB free"
    except Exception:
        return None


def _cpu_name() -> str:
    if platform.system() == "Linux":
        with contextlib.suppress(Exception):
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _nvidia_driver() -> Optional[str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except Exception:
        return None


def _torch_lines() -> list:
    try:
        import torch
    except Exception as exc:
        return [f"torch             : not importable ({exc.__class__.__name__}: {exc})"]
    lines = [f"torch             : {torch.__version__}"]
    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    backend = "rocm" if hip else "cuda" if cuda else "cpu-only build"
    lines.append(f"torch backend     : {backend} (cuda={cuda}, hip={hip})")
    try:
        available = torch.cuda.is_available()
    except Exception as exc:  # broken driver etc.
        lines.append(f"gpu available     : error ({exc})")
        return lines
    lines.append(f"gpu available     : {available}")
    if available:
        with contextlib.suppress(Exception):
            lines.append(f"cudnn             : {torch.backends.cudnn.version()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            lines.append(f"gpu[{i}]            : {p.name} | sm_{p.major}{p.minor} | {p.total_memory / 2**30:.1f} GB")
        drv = _nvidia_driver()
        if drv:
            lines.append(f"nvidia driver     : {drv}")
    return lines


def _audio_lines() -> list:
    try:
        import sounddevice as sd
    except Exception as exc:
        return [f"sounddevice       : unavailable ({exc.__class__.__name__}: {exc})"]
    lines = []
    try:
        apis = [a["name"] for a in sd.query_hostapis()]
        lines.append(f"audio host APIs   : {', '.join(apis)}")
        default_in, default_out = sd.default.device
        for label, idx in (("default input", default_in), ("default output", default_out)):
            if idx is not None and idx >= 0:
                d = sd.query_devices(idx)
                lines.append(f"{label:<17} : [{idx}] {d['name']} @ {int(d['default_samplerate'])} Hz "
                             f"({apis[d['hostapi']]})")
    except Exception as exc:
        lines.append(f"audio query       : failed ({exc})")
    return lines


def system_info_lines() -> list:
    lines = [
        f"session started   : {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"platform          : {platform.platform()}",
        f"python            : {platform.python_version()} ({sys.executable})",
        f"cwd               : {os.getcwd()}",
        f"cpu               : {_cpu_name()} | {os.cpu_count()} logical cores",
        f"ram               : {_ram_gb() or 'unknown'}",
    ]
    lines += _torch_lines()
    lines += _audio_lines()
    versions = {p: _pkg_version(p) for p in _PACKAGES}
    lines.append("packages          : " + ", ".join(f"{k}={v}" for k, v in versions.items() if v))
    env = {k: os.environ[k] for k in _ENV_VARS if k in os.environ}
    if env:
        lines.append("environment       : " + ", ".join(f"{k}={v}" for k, v in env.items()))
    return lines


def log_system_info(logger: Optional[logging.Logger] = None) -> None:
    logger = logger or get_logger("system")
    logger.info("=" * 78)
    for line in system_info_lines():
        logger.info(line)
    logger.info("=" * 78)


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def _prune(log_dir: Path, keep: int) -> None:
    files = sorted(log_dir.glob("session_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        with contextlib.suppress(OSError):
            old.unlink()


def _install_exception_hooks() -> None:
    logger = get_logger("crash")

    def _sys_hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    def _thread_hook(args):
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread else "?"
        logger.critical("Uncaught exception in thread %s", name,
                        exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook


def _install_qt_handler() -> None:
    try:
        from PyQt6.QtCore import QtMsgType, qInstallMessageHandler
    except Exception:
        return
    logger = get_logger("qt")
    levels = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }
    qInstallMessageHandler(lambda mode, ctx, msg: logger.log(levels.get(mode, logging.INFO), msg))


def setup_logging(log_dir, *, console_level: int = logging.INFO, file_level: int = logging.DEBUG,
                  keep: int = 20, capture_qt: bool = True) -> Path:
    """Create ./logs/session_<timestamp>.log, attach handlers, log system info. Idempotent."""
    global _configured, _log_path
    if _configured and _log_path is not None:
        return _log_path

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    _prune(log_dir, keep - 1)  # leave room for the file we are about to create

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"session_{stamp}.log"
    n = 1
    while path.exists():  # two launches within one second
        n += 1
        path = log_dir / f"session_{stamp}_{n}.log"

    formatter = logging.Formatter(FMT, DATEFMT)
    root = logging.getLogger(ROOT_LOGGER)
    root.setLevel(min(console_level, file_level))
    root.propagate = False

    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(console_level)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # third-party chatter and Python warnings go to the file too
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").addHandler(fh)

    _install_exception_hooks()
    if capture_qt:
        _install_qt_handler()

    _configured, _log_path = True, path
    log_system_info()
    get_logger("log").info("Log file: %s", path)
    return path


# --------------------------------------------------------------------------- #
# Small helpers for hot paths
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def timed(logger: logging.Logger, label: str, level: int = logging.DEBUG):
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.log(level, "%s took %.1f ms", label, (time.perf_counter() - start) * 1000.0)


class RealtimeStats:
    """Collects per-block timings; report() gives one compact line every few seconds.

    RTF = processing time / audio time. RTF >= 1.0 means the stream cannot keep up
    (input overflows / output underruns).
    """

    def __init__(self, logger: Optional[logging.Logger] = None, every_s: float = 5.0, window: int = 200):
        self.log = logger or get_logger("realtime")
        self.every_s = every_s
        self._infer = deque(maxlen=window)
        self._block_s = 0.0
        self._blocks = 0
        self._skipped = 0
        self._late = 0
        self._underruns = 0
        self._dropped = 0
        self._last_report = time.monotonic()

    def record(self, infer_s: float, block_s: float) -> None:
        self._infer.append(infer_s)
        self._block_s = block_s
        self._blocks += 1
        if infer_s > block_s:
            self._late += 1
            self.log.warning("Block took %.0f ms but only %.0f ms of audio - stream will glitch",
                             infer_s * 1000, block_s * 1000)

    def skipped(self) -> None:
        self._skipped += 1

    def underrun(self) -> None:
        """Output callback had nothing to play (audible gap)."""
        self._underruns += 1

    def dropped(self) -> None:
        """Input was discarded because inference fell too far behind."""
        self._dropped += 1

    def report(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_report < self.every_s:
            return
        self._last_report = now
        if not self._infer:
            return
        xs = sorted(self._infer)
        mean = sum(xs) / len(xs)
        p95 = xs[min(len(xs) - 1, int(len(xs) * 0.95))]
        rtf = mean / self._block_s if self._block_s else float("nan")
        self.log.info("blocks=%d idle=%d late=%d underruns=%d dropped=%d | infer mean=%.0f ms "
                      "p95=%.0f ms | block=%.0f ms | RTF=%.2f",
                      self._blocks, self._skipped, self._late, self._underruns, self._dropped,
                      mean * 1000, p95 * 1000, self._block_s * 1000, rtf)