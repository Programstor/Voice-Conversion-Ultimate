"""
check_runtime.py - verify that the freshly built runtime can actually run the app.

Run by setup_linux.sh and setup_win.bat with the runtime's own Python:
    runtime_linux/bin/python check_runtime.py
Exit code 0 = everything the app needs works, 1 = at least one [FAIL].

torch.cuda.is_available() alone is not proof that the GPU works (a wheel built for the wrong
architecture still reports True), so the GPU check runs a real kernel.
"""
from __future__ import annotations

import platform
import sys

failures = []


def check(name, fn, hint=""):
    try:
        print(f"  [ ok ] {name:<16} {fn()}")
    except Exception as exc:  # noqa: BLE001 - we want to report anything
        failures.append(name)
        print(f"  [FAIL] {name:<16} {type(exc).__name__}: {exc}")
        if hint:
            print(f"         -> {hint}")


def torch_info():
    import torch

    if torch.cuda.is_available():
        text = f"{torch.__version__} (built for CUDA {torch.version.cuda})"
        major, minor = torch.cuda.get_device_capability(0)
        x = torch.randn(512, 512, device="cuda")
        float((x @ x).sum().item())            # real kernel launch, not just a flag
        text += f" | GPU: {torch.cuda.get_device_name(0)} (sm_{major}{minor}), kernel launch OK"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        text = f"{torch.__version__} (macOS build)"
        x = torch.randn(512, 512, device="mps")
        float((x @ x).sum().item())            # real kernel launch on MPS
        text += " | MPS kernel launch OK"
    else:
        text = f"{torch.__version__}"
        text += " | no usable GPU - CPU mode"
    return text


def version_of(module_name, attr="__version__"):
    def _inner():
        module = __import__(module_name)
        return getattr(module, attr, "import OK")
    return _inner


def qt_info():
    from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
    return f"PyQt {PYQT_VERSION_STR} / Qt {QT_VERSION_STR}"


def audio_info():
    import sounddevice as sd
    apis = ", ".join(a["name"] for a in sd.query_hostapis())
    return f"host APIs: {apis or 'none found'}"


def sndfile_info():
    import soundfile as sf
    return f"libsndfile {sf.__libsndfile_version__}"


def faiss_info():
    import faiss
    v = getattr(faiss, "__version__", None) or getattr(faiss, "version", "import OK")
    gpu = hasattr(faiss, "StandardGpuResources")
    return f"{v} ({'gpu' if gpu else 'cpu'})"


def main():
    print(f"Python {platform.python_version()} on {platform.platform()}")
    check("torch", torch_info)
    check("torchaudio", version_of("torchaudio"))
    check("transformers", version_of("transformers"))
    check("torchfcpe", version_of("torchfcpe"))
    check("faiss", faiss_info)
    check("numpy", version_of("numpy"))
    check("scipy", version_of("scipy"))
    check("librosa", version_of("librosa"))
    check("soundfile", sndfile_info)
    check("sounddevice", audio_info,
          hint="PortAudio is missing. Fedora: sudo dnf install portaudio | "
               "Debian/Ubuntu: sudo apt install libportaudio2")
    check("PyQt6", qt_info,
          hint="Linux: if the app later fails to start with an 'xcb' error, install the Qt xcb "
               "libraries (Fedora: sudo dnf install xcb-util-cursor)")
    check("pyqtgraph", version_of("pyqtgraph"))
    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())