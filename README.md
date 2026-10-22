<h1 align="center">Voice Conversion Ultimate</h1>

<p align="center">
  <img alt="Github top language" src="https://img.shields.io/github/languages/top/Programstor/Voice-Conversion-Ultimate?color=56BEB8">
  <img alt="Repository size" src="https://img.shields.io/github/repo-size/Programstor/Voice-Conversion-Ultimate?color=56BEB8">
  <img alt="License" src="https://img.shields.io/github/license/Programstor/Voice-Conversion-Ultimate?color=56BEB8">
</p>

<p align="center">
  <a href="#about">About</a> &#xa0; | &#xa0;
  <a href="#requirements">Requirements</a> &#xa0; | &#xa0;
  <a href="#setup">Setup</a> &#xa0; | &#xa0;
  <a href="#app-guide">App Guide</a> &#xa0; | &#xa0;
  <a href="#config">Config</a> &#xa0; | &#xa0;
  <a href="#to-do">To Do</a> &#xa0; | &#xa0;
  <a href="#license">License</a> &#xa0; | &#xa0;
  <a href="https://github.com/Programstor" target="_blank">Author</a>
</p>

<br>

## About

Voice Conversion Ultimate is a modern version of the [Retrieval-based Voice Conversion App](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) with a shared inference / realtime pipeline, bundled with serious optimizations and truncations, saving you time and pain during voice / audio inference and training.

All this, with **full CPU support, NVIDIA CUDA, AMD ROCm on Linux, and Apple MPS on supported macOS systems**, bundled into a single, sleek and easy-to-use PyQt 6 app.

**Windows and Linux are the primary tested platforms. Linux is recommended for AMD GPU users because ROCm support is Linux-only. macOS is supported through PyTorch's MPS backend on compatible systems, but macOS training is currently not tested.**

## Requirements

### Python

Python **3.12**.

**Python 3.11 and 3.13 are currently not supported by the project.**

### GPU — NVIDIA (CUDA)
Any NVIDIA GPU with **CUDA compute capability 6.1 or higher** (Pascal / GTX 1000 series and later). The launcher auto-detects your driver and selects the appropriate CUDA wheel (`cu126` for drivers 525–579, `cu130` for 580+). Older drivers fall back to CPU, so make sure you have the latest for your card!

The `faiss-gpu` (Facebook AI Similarity Search) library is automatically installed on Linux systems for moderately faster index retrieval. It is **not available on Windows** — the app falls back to `faiss-cpu` if so.

### GPU — AMD (ROCm, Linux only)
ROCm **6.x** on Linux. Supported cards are **RDNA 2 (RX 6000 series) and newer**. Older architectures (Vega, Polaris) are not supported by the ROCm PyTorch wheels. The launcher reads the installed ROCm version from `/opt/rocm` and selects the matching wheel tag automatically.

`faiss-gpu` has no ROCm wheels yet, so `faiss-cpu` is installed!

**ROCm is not supported on Windows.** AMD GPU users on Windows will run CPU-only.

### GPU — Apple (MPS, macOS)

Supported macOS systems can use PyTorch's **MPS** backend through Apple's Metal framework.

MPS is used automatically when available. On systems where MPS is unavailable, the application falls back to CPU execution.

On macOS, `launch_lin.sh` installs the standard PyTorch macOS build and does **not** use the CUDA/ROCm backend selector.

**macOS training is currently not tested.**

### CPU

Any **x86-64 CPU with at least AVX2** is supported (Intel Haswell / AMD Ryzen 1000 or later).

Inference works on x86-64 systems without a GPU; performance is simply lower than with a supported accelerator.

The application detects available CPU instruction-set features and selects the best compatible dtype / execution path automatically.

ARM Linux is not currently tested or supported.

### C++ compiler (CPU training only)

A C++ compiler is only required if you intend to **train on CPU**. Inference never uses `torch.compile` and runs without any compiler on all platforms.

When training on CPU, TorchInductor compiles the generator and discriminator forward passes to native C++ for a meaningful speed-up. If no compiler is found, the app logs a notice and trains in eager mode — slower, but fully functional.

**Windows — Visual Studio with Desktop C++ workload**

1. Download and run the [Visual Studio 2026 installer](https://visualstudio.microsoft.com/downloads/).
2. In the installer, select the **"Desktop development with C++"** workload.
3. Complete the installation — the launcher configures the compiler automatically on every launch.

**Linux:** `g++` — `sudo apt install build-essential` (Debian/Ubuntu) or `sudo dnf install gcc-c++` (Fedora).

**macOS:** **Xcode Command Line Tools** — `xcode-select --install`. Apple's `clang++` is sufficient.

## Setup

Open the corresponding launcher:

* **Windows:** `launch_win.bat`
* **Linux / macOS:** `launch_lin.sh`

The launchers handle the setup automatically. They:

1. Install `uv` if necessary.
2. Create an isolated Python runtime.
3. Detect the GPU / CPU environment.
4. Install the appropriate PyTorch build.
5. Install the required project dependencies.
6. Verify the runtime before launching the application.

You normally do not need to install Python, PyTorch, CUDA or the project dependencies manually.

### 1. Get the repository

As this app is currently in its **beta phase**, you can get the full app bundle by:

* Cloning it:

```bash
git clone https://github.com/Programstor/Voice-Conversion-Ultimate

cd Voice-Conversion-Ultimate
```

* Downloading an archive and unpacking it manually:

`Code (top right of the file list) -> Download ZIP`

### 2. Download the pretrained models

Because of GitHub limitations, you will have to manually download the pretrained `.pth` models for training.

**Recommended:** the more modern **TITAN pretrained models**.

Go to [TITAN Pretrains on HuggingFace](https://huggingface.co/blaise-tk/TITAN/tree/main/models/medium) and, from the `40k/pretrained/` and `48k/pretrained/` folders, download:

```text
G-f040k-TITAN-Medium.pth
D-f040k-TITAN-Medium.pth

G-f048k-TITAN-Medium.pth
D-f048k-TITAN-Medium.pth
```

To get the **official pitch-aware (f0) RVC v2 pretrained models**, go to [RVC-WebUI on HuggingFace](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/pretrained_v2) and download:

```text
f0G40k.pth
f0D40k.pth

f0G48k.pth
f0D48k.pth
```

Place all pretrained models in:

```text
assets/pretrained/
```

These files must be present before you run **Model Training**.

### 3. Launch

**Windows:**

```bat
launch_win.bat
```

**Linux / macOS:**

```bash
bash launch_lin.sh
```

On first run, the launcher creates the runtime and installs the required packages. This can take a few minutes.

Subsequent launches skip setup unless:

* the runtime is missing or invalid;
* `requirements.txt` has changed;
* `RECREATE=1` is specified; or
* the detected GPU / PyTorch backend no longer matches the runtime that was previously installed.

You can override the detected backend at any time.

**Linux:**

```bash
TORCH_BACKEND=cpu bash launch_lin.sh
```

**Windows:**

```bat
set "TORCH_BACKEND=cpu" && launch_win.bat
```

For Windows CUDA, the valid backend values are currently:

```text
cu130
cu126
cpu
```

For macOS, the normal launcher automatically installs the standard PyTorch macOS build and uses **MPS when available**. You normally do not need to set a backend manually.

To force a full runtime reinstall:

**Linux / macOS:**

```bash
RECREATE=1 bash launch_lin.sh
```

**Windows:**

```bat
set "RECREATE=1" && launch_win.bat
```

## App Guide

### Inference Tab

Convert an audio file to a target voice. Select a voice model (`.pth`) and an index file (`.index`) — the app auto-matches an index to the selected model by name.

Browse for a source audio file (vocal track) and click **Infer Audio**. The result plays back in the built-in audio player.

Controls:

* **Transpose** — shift the output pitch up or down in semitones. Positive = higher, negative = lower. `0` = no shift.
* **Volume Scaling** — RMS envelope matching strength. `100%` = output level tracks the source; `0%` = use the model's own output level.
* **Voiceless Protection** — fraction of breath and consonant frames that bypass index retrieval and retain the original features. Lower values protect sibilants and breaths more aggressively.
* **Index Rate** — how strongly the retrieval index pulls the timbre toward the training data. Higher values make the result more characteristic of the target voice, but can introduce artifacts when matching is poor.
* **Resample to** — optionally resample the output to a standard rate before saving. Set to `0` to keep the model's native sample rate.
* **Stitch Instruments** — when checked, instruments separated from the source are mixed back into the converted vocal output.
* **Keep Stereo** — applies the built-in pseudo-stereo / room-style stereo effect to the converted output for a wider and more natural presentation. Set its strength to your liking in the config.

### Training Tab

Train a new voice model from scratch using your own recordings.

#### Model Options

* **Model name** — identifier for this run. All outputs (checkpoints, logs, index) are saved under `user/training/<name>/`. You must retype the same name later if you want to resume training from a checkpoint or train an index.
* **Training folder** — folder containing your raw audio dataset. Defaults to `user/dataset/`.
* **Sample rate** — 40 kHz or 48 kHz. Must match the pretrained weights you intend to use. Choose before preprocessing; it cannot be changed mid-run.

#### Training Options

* **Total Epochs** — number of full passes through the dataset. More epochs generally improve convergence, with diminishing returns and eventual overfitting. `50–200` is typical depending on dataset size and quality.
* **Save Frequency** — a checkpoint is written every this many epochs.
* **Batch Size** — number of audio chunks processed per gradient step. Increase if you have VRAM headroom; decrease if you get out-of-memory errors.

#### Buttons

* **Preprocess** — slice, resample, and extract ContentVec features and F0 curves from the training folder. Must be run before training.
* **Train Model** — starts the training loop. The loss chart updates live during training.
* **Train Index** — builds a FAISS retrieval index from the extracted features after training. Use the resulting `.index` file alongside the `.pth` for best inference quality.
* **Sequence** — runs Preprocess, Train Model, and Train Index sequentially and automatically resumes if the training is stopped.

Advanced training settings such as AMP mode, CPU threads, dataset caching, and TorchInductor compilation are controlled through `config.toml` — see [Config](#config).

**CPU training note:** when training on CPU with a compatible C++ compiler present, TorchInductor automatically compiles the generator and discriminator on the first epoch for faster subsequent steps. The first epoch will be slower while compilation happens. See [C++ compiler](#c-compiler-cpu-training-only) for setup.

### Realtime Tab

Live voice conversion through your microphone, routed to an output device.

Select a voice model, index, microphone, and speaker, then click **Infer Voice** to start the stream. The button is a toggle — click again to stop.

Controls:

* **Transpose, Volume Scaling, Voiceless Protection, Index Rate** — same meaning as in Inference. All four update live while the stream is running without restarting it.
* **Block Length** — audio chunk converted per inference step, in milliseconds. This is the primary latency control. Lower values reduce latency but increase CPU/GPU load and the risk of dropout. `200–500 ms` is a reasonable starting range.
* **Noise Gate** — threshold below which a block is treated as silence and passed through without inference, saving computation during quiet moments.

## Config

Defaults for DSP, realtime, index, and training can be overridden without modifying source code by placing a `config.toml` in the project root.

Only the keys you specify are changed; everything else keeps its default.

*This is a demonstration of the config format, not a recommended configuration.*

```toml
[realtime]

block_ms = 250
crossfade_ms = 40

[index]

max_vectors = 20000
top_k = 8

[train]

amp = "bf16"
cpu_threads = 8
epochs = 100
cache_specs = true

[dsp]

gate_db = -40.0
highpass_hz = 60.0
```

Supported sections are:

```text
[dsp]
[realtime]
[index]
[train]
[inference]
```

Unknown keys are logged as warnings and ignored.

## To Do

### Features planned for the Beta phase

`Torch compilation support for GPU training as well, using Cudagraphs to speed up the already faster training process`

`PyMsss Integration - for in-app vocal / instrumental separation, which is now done with external apps like UVR or some online tools`

`ONNX Runtime Exports - for even faster CPU inference for users without access to advanced hardware or a GPU in general`

`Extended Model Compatibility - support for inference and training of 32k models and older text-only (nof0) models`

`In-App Customization - make working with files easier and more integrated into the app, including configuration, themes and model exploring`

### Features planned for after Release

`Diffusion Transform - state-of-the-art inference, requiring no preprocessing or model training, to produce a closer vocal and timbre match with the original speaker`

`BigVGAN v2 Vocoder - uses Snake activations, which may provide different results from the current VITS architecture's LeakyReLU activations. This is part of the planned DiT expansion.`

## License

This project is under the MIT license. For more details see the [LICENSE](LICENSE) file.

Made with love by <a href="https://github.com/Programstor" target="_blank">Programstor</a>

 

<a href="#top">Back to top</a>