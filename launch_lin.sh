#!/bin/bash
# Voice Conversion Ultimate - Linux & macOS Launcher & Setup
# Backends: CUDA (cu130/cu126), ROCm (Linux only), MPS (macOS Apple Silicon), CPU

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 1
export VCU_HOME="$ROOT_DIR"

# ------------------------------------------------------------------------------
# 1. Terminal Auto-Spawn & Pause Setup
# ------------------------------------------------------------------------------
if [ -z "${VCU_SPAWNED:-}" ] && ! [ -t 0 ] && ! [ -t 1 ] && ! [ -t 2 ]; then
    script="$(readlink -f "${BASH_SOURCE[0]}")"
    run=(env VCU_SPAWNED=1 bash "$script" "$@")

    candidates=()
    [ -n "${TERMINAL:-}" ] && candidates+=("$TERMINAL")
    candidates+=(konsole gnome-terminal xfce4-terminal alacritty kitty foot wezterm lxterminal xterm)

    for emu in "${candidates[@]}"; do
        command -v "$emu" >/dev/null 2>&1 || continue
        case "$(basename "$emu")" in
            konsole|alacritty|xterm|lxterminal) exec "$emu" -e "${run[@]}" ;;
            gnome-terminal)                      exec "$emu" -- "${run[@]}" ;;
            xfce4-terminal)                      exec "$emu" -x "${run[@]}" ;;
            wezterm)                             exec "$emu" start -- "${run[@]}" ;;
            kitty|foot)                          exec "$emu" "${run[@]}" ;;
            *)                                   exec "$emu" -e "${run[@]}" ;;
        esac
    done

    msg="No terminal emulator found. Please run this script from a terminal."
    echo "$msg" >&2
    if   command -v kdialog     >/dev/null 2>&1; then kdialog --error "$msg"
    elif command -v zenity      >/dev/null 2>&1; then zenity --error --text="$msg"
    elif command -v notify-send >/dev/null 2>&1; then notify-send "Voice Conversion Ultimate" "$msg"
    fi
    exit 1
fi

if [ -n "${VCU_SPAWNED:-}" ] && [ -z "${VCU_NO_PAUSE:-}" ]; then
    trap 'code=$?; sleep 0.3; if ( : </dev/tty ) 2>/dev/null; then echo; [ "$code" -eq 0 ] && echo "Finished." || echo "Stopped with exit code $code."; read -rp "Press Enter to close this window..." _; fi' EXIT
fi

# ------------------------------------------------------------------------------
# 2. Paths & Environment Variables
# ------------------------------------------------------------------------------
if [[ "$(uname)" == "Darwin" ]]; then
    RUNTIME_DIR="$ROOT_DIR/runtime_mac"
else
    RUNTIME_DIR="$ROOT_DIR/runtime_linux"
fi
PYTHON_EXE="$RUNTIME_DIR/bin/python"
REQ_FILE="$ROOT_DIR/requirements.txt"
CHECK_SCRIPT="$ROOT_DIR/check_runtime.py"
MARKER="$RUNTIME_DIR/.setup_complete"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/setup_linux.log"

setup_fail() {
    echo
    echo "Setup failed - see $LOG_FILE"
    exit 1
}

# ------------------------------------------------------------------------------
# 3. Backend mismatch check — handle overrides vs actual environment changes
# ------------------------------------------------------------------------------
_USER_OVERRIDE="${TORCH_BACKEND:-}"

if [ -z "${TORCH_BACKEND:-}" ]; then
    if [[ "$(uname)" == "Darwin" ]]; then
        if [[ "$(uname -m)" == "arm64" ]]; then
            _DETECTED_BACKEND=cpu_mps
        else
            _DETECTED_BACKEND=cpu_intel
        fi
    else
        _DRIVER_MAJOR=""
        if command -v nvidia-smi >/dev/null 2>&1; then
            _DV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')"
            _DRIVER_MAJOR="${_DV%%.*}"
        fi

        if [[ "$_DRIVER_MAJOR" =~ ^[0-9]+$ ]]; then
            if   [ "$_DRIVER_MAJOR" -ge 580 ]; then _DETECTED_BACKEND=cu130
            elif [ "$_DRIVER_MAJOR" -ge 525 ]; then _DETECTED_BACKEND=cu126
            else _DETECTED_BACKEND=cpu
            fi
        elif command -v rocminfo >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1; then
            _RV=""
            [ -f /opt/rocm/.info/version ] && _RV="$(cat /opt/rocm/.info/version | tr -d '[:space:]')"
            [ -z "$_RV" ] && command -v rocminfo >/dev/null 2>&1 && \
                _RV="$(rocminfo 2>/dev/null | awk '/ROCm version/ {print $NF; exit}')"
            _RM="${_RV%%.*}"
            _Rm="$(echo "$_RV" | cut -d. -f2)"
            if   [ "${_RM:-0}" -ge 6 ] && [ "${_Rm:-0}" -ge 3 ]; then _DETECTED_BACKEND=rocm6.3
            elif [ "${_RM:-0}" -ge 6 ] && [ "${_Rm:-0}" -ge 2 ]; then _DETECTED_BACKEND=rocm6.2
            elif [ "${_RM:-0}" -ge 6 ];                           then _DETECTED_BACKEND=rocm6.1
            else _DETECTED_BACKEND=rocm5.7
            fi
        else
            _DETECTED_BACKEND=cpu
        fi
    fi
else
    _DETECTED_BACKEND="$TORCH_BACKEND"
fi

# Compare against what was used when the runtime was last built.
if [ -f "$MARKER" ]; then
    _STORED_BACKEND="$(grep '^backend=' "$MARKER" 2>/dev/null | cut -d= -f2)"
    if [ -n "$_STORED_BACKEND" ] && [ "$_STORED_BACKEND" != "$_DETECTED_BACKEND" ]; then
        if [ -n "$_USER_OVERRIDE" ]; then
            echo "  [Notice] TORCH_BACKEND='$_USER_OVERRIDE' override detected (runtime was built for '$_STORED_BACKEND')."
            echo "           Bypassing automatic reinstall for debugging session."
        else
            echo "  GPU environment changed: runtime was built for '$_STORED_BACKEND', now '$_DETECTED_BACKEND'."
            echo "  Triggering reinstall..."
            RECREATE=1
        fi
    fi
fi

# ------------------------------------------------------------------------------
# 4. Setup Logic
# ------------------------------------------------------------------------------
if [ ! -f "$MARKER" ] || [ ! -x "$PYTHON_EXE" ] || [ "$REQ_FILE" -nt "$MARKER" ] || [ "${RECREATE:-0}" = "1" ]; then
    mkdir -p "$LOG_DIR"
    exec > >(tee -a "$LOG_FILE") 2>&1

    echo "=== Voice Conversion Ultimate - Linux Runtime Setup ==="
    echo "Log:     $LOG_FILE"
    echo "Started: $(date '+%a %m/%d/%Y %H:%M:%S.%N' | cut -c1-23)"
    echo

    echo "[1/5] Checking prerequisites..."
    if [ ! -f "$REQ_FILE" ]; then
        echo "ERROR: requirements.txt not found: $REQ_FILE"
        setup_fail
    fi
    if [ ! -f "$CHECK_SCRIPT" ]; then
        echo "ERROR: check_runtime.py not found: $CHECK_SCRIPT"
        setup_fail
    fi

    LDCONFIG="$(command -v ldconfig || echo /sbin/ldconfig)"
    if ! "$LDCONFIG" -p 2>/dev/null | grep -q 'libportaudio\.so'; then
        echo "  WARNING: PortAudio not found (sounddevice needs it for audio in/out)."
        echo "           Fedora: sudo dnf install portaudio    Debian/Ubuntu: sudo apt install libportaudio2"
    fi

    if [ -z "${TORCH_BACKEND:-}" ]; then
        if [[ "$(uname)" == "Darwin" ]]; then
            TORCH_BACKEND=cpu
            if [[ "$(uname -m)" == "arm64" ]]; then
                export TORCH_DEVICE="${TORCH_DEVICE:-mps}"
                echo "  Detected Apple Silicon ($(uname -m)) -- MPS backend available at runtime."
                echo "  TORCH_BACKEND=cpu (standard wheel includes MPS), TORCH_DEVICE=mps"
            else
                export TORCH_DEVICE="${TORCH_DEVICE:-cpu}"
                echo "  Detected Intel Mac -- MPS not available, using CPU."
                echo "  TORCH_BACKEND=cpu, TORCH_DEVICE=cpu"
            fi
            echo "  Override device with:  TORCH_DEVICE=cpu bash launch_lin.sh"
        else
            DRIVER_VERSION=""
            DRIVER_MAJOR=""
            if command -v nvidia-smi >/dev/null 2>&1; then
                DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')"
                DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
            fi

            if [[ "$DRIVER_MAJOR" =~ ^[0-9]+$ ]]; then
                if   [ "$DRIVER_MAJOR" -ge 580 ]; then TORCH_BACKEND=cu130
                elif [ "$DRIVER_MAJOR" -ge 525 ]; then TORCH_BACKEND=cu126
                else
                    TORCH_BACKEND=cpu
                    echo "  WARNING: NVIDIA driver $DRIVER_VERSION is too old for CUDA - using CPU."
                fi
                echo "  Detected NVIDIA GPU -- TORCH_BACKEND=$TORCH_BACKEND [driver: $DRIVER_VERSION]"

            elif command -v rocminfo >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1; then
                ROCM_VERSION=""
                if [ -f /opt/rocm/.info/version ]; then
                    ROCM_VERSION="$(cat /opt/rocm/.info/version | tr -d '[:space:]')"
                elif command -v rocminfo >/dev/null 2>&1; then
                    ROCM_VERSION="$(rocminfo 2>/dev/null | awk '/ROCm version/ {print $NF; exit}')"
                fi
                ROCM_MAJOR="${ROCM_VERSION%%.*}"
                ROCM_MINOR="$(echo "$ROCM_VERSION" | cut -d. -f2)"

                if   [ "${ROCM_MAJOR:-0}" -ge 6 ] && [ "${ROCM_MINOR:-0}" -ge 3 ]; then
                    TORCH_BACKEND=rocm6.3
                elif [ "${ROCM_MAJOR:-0}" -ge 6 ] && [ "${ROCM_MINOR:-0}" -ge 2 ]; then
                    TORCH_BACKEND=rocm6.2
                elif [ "${ROCM_MAJOR:-0}" -ge 6 ]; then
                    TORCH_BACKEND=rocm6.1
                else
                    TORCH_BACKEND=rocm5.7
                    echo "  WARNING: ROCm ${ROCM_VERSION:-unknown} is old; rocm5.7 wheel selected."
                    echo "           Upgrading to ROCm 6.x is strongly recommended."
                fi
                echo "  Detected AMD GPU (ROCm ${ROCM_VERSION:-unknown}) -- TORCH_BACKEND=$TORCH_BACKEND"
                echo "  NOTE: ROCm is Linux-only. torch.cuda.* APIs map transparently to HIP."

            else
                TORCH_BACKEND=cpu
                echo "  No supported GPU found - using CPU build of PyTorch."
                echo "  Training will work but is very slow. Inference is usable."
            fi

            echo "  Override any time with:  TORCH_BACKEND=<tag> bash launch_lin.sh"
        fi
    else
        echo "  TORCH_BACKEND=$TORCH_BACKEND [from environment]"
    fi

    echo
    echo "[2/5] Checking for uv..."
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        command -v curl >/dev/null 2>&1 || { echo "ERROR: curl is required to install uv."; setup_fail; }
        echo "  Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "ERROR: uv could not be installed or is not on PATH."; setup_fail; }
        export PATH="$HOME/.local/bin:$PATH"
        command -v uv >/dev/null 2>&1 || { echo "ERROR: uv could not be installed or is not on PATH."; setup_fail; }
    fi
    echo "  $(uv --version)"

    echo
    echo "[3/5] Preparing Python $PYTHON_VERSION runtime at $RUNTIME_DIR ..."
    if [ "${RECREATE:-0}" = "1" ] && [ -d "$RUNTIME_DIR" ]; then
        echo "  RECREATE=1 - removing existing runtime."
        
        if [ "$RUNTIME_DIR" = "$ROOT_DIR" ]; then
            echo "ERROR: RUNTIME_DIR equals ROOT_DIR - aborting to prevent data loss."
            setup_fail
        fi
        rm -rf "$RUNTIME_DIR"
    fi

    if [ -x "$PYTHON_EXE" ]; then
        echo "  Reusing existing runtime."
    else
        uv venv "$RUNTIME_DIR" --python "$PYTHON_VERSION" --seed || { echo "ERROR: Could not create runtime."; setup_fail; }
    fi
    rm -f "$MARKER"

    echo
    echo "[4/5] Installing packages from requirements.txt - PyTorch backend: $TORCH_BACKEND ..."
    uv pip install --python "$PYTHON_EXE" --torch-backend "$TORCH_BACKEND" -r "$REQ_FILE" \
        || { echo; echo "ERROR: Package installation failed. Run 'uv self update' if error mentions --torch-backend."; setup_fail; }

    if [[ "$(uname)" != "Darwin" ]]; then
        case "$TORCH_BACKEND" in
            cu*)
                uv pip install --python "$PYTHON_EXE" faiss-gpu \
                    || { echo "  WARNING: faiss-gpu failed, installing faiss-cpu instead."
                         uv pip install --python "$PYTHON_EXE" faiss-cpu; }
                ;;
            *)
                uv pip install --python "$PYTHON_EXE" faiss-cpu
                ;;
        esac
    fi

    echo
    echo "[5/5] Verifying runtime..."
    "$PYTHON_EXE" "$CHECK_SCRIPT" || { echo; echo "ERROR: Verification failed. Fix [FAIL] items above and rerun."; setup_fail; }

    if [[ "$(uname)" == "Darwin" ]]; then
        if [[ "$(uname -m)" == "arm64" ]]; then _MARKER_BACKEND=cpu_mps
        else _MARKER_BACKEND=cpu_intel
        fi
    else
        _MARKER_BACKEND="$TORCH_BACKEND"
    fi
    printf 'backend=%s\ndate=%s\n' "$_MARKER_BACKEND" "$(date '+%a %m/%d/%Y %H:%M:%S.%N' | cut -c1-23)" > "$MARKER"
    
    echo
    echo "=== Setup complete ==="
    echo "Finished: $(date '+%a %m/%d/%Y %H:%M:%S.%N' | cut -c1-23)"
    echo "Runtime:  $RUNTIME_DIR"
    echo
fi

# ------------------------------------------------------------------------------
# 5. Launch App
# ------------------------------------------------------------------------------
exec >/dev/tty 2>&1

# Ensure the backend choice is explicitly exported to Python
export TORCH_BACKEND="$_DETECTED_BACKEND"

# If CPU mode is active (either natively or via override), mask CUDA devices
if [ "$TORCH_BACKEND" = "cpu" ]; then
    export CUDA_VISIBLE_DEVICES="-1"
    unset CUDA_HOME
    unset CUDA_PATH
fi

echo "Launching Voice Conversion Ultimate ..."
# TORCH_LOGS, TORCH_COMPILE_DEBUG, and other outer shell variables 
# automatically flow through to python execution here.
"$PYTHON_EXE" "$ROOT_DIR/app.py" "$@"