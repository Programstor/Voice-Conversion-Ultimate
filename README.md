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
  <a href="#tabs">App Guide</a> &#xa0; | &#xa0;
  <a href="#config">Config</a> &#xa0; | &#xa0;
  <a href="#todo">To Do</a> &#xa0; | &#xa0;
  <a href="#license">License</a> &#xa0; | &#xa0;
  <a href="https://github.com/Programstor" target="_blank">Author</a>
</p>

<br>

## About

Voice Conversion Ultimate is a modern version of the [Retrieval-based Voice Conversion App](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) with a shared inference / realtime pipeline, bundled with some serious optimizations and truncations, saving you the time and pain during voice / audio inference and training. All this, with FULL CPU support, CUDA and AMD ROCm (Linux only!), bundled in a single, sleek and easy to use PyQt 6 app! **Use on Windows or Linux for best results, especially Linux for AMD cards! MacOs is generally supported (not tested for training), but will use 'cpu' as a backend!**

## Requirements

### Python
Python **3.11** or **3.12**. **3.13** is not yet supported (PyTorch wheels are not released yet).

### GPU — NVIDIA (CUDA)
Any NVIDIA GPU with **CUDA compute capability 6.1 or higher** (Pascal / GTX 1000 series and later). The launcher auto-detects your driver and selects the appropriate CUDA wheel (`cu126` for drivers 525–579, `cu130` for 580+). Older drivers fall back to CPU, so make sure you have the latest for your card!

The `faiss-gpu` (Facebook AI Similarity Search) library is automatically installed on Linux systems for moderately faster index retrieval. It is **not available on Windows** — the app falls back to `faiss-cpu` if so.

### GPU — AMD (ROCm, Linux only)
ROCm **6.x** on Linux. Supported cards are **RDNA 2 (RX 6000 series) and newer**. Older architectures (Vega, Polaris) are not supported by the ROCm PyTorch wheels. The launcher reads the installed ROCm version from `/opt/rocm` and selects the matching wheel tag automatically.

`faiss-gpu` has no ROCm wheels yet, so `faiss-cpu` is installed!

**ROCm is not supported on Windows.** AMD GPU users on Windows will run CPU-only.

### CPU
Any **x86-64** CPU with at least AVX2 (Intel Haswell / AMD Ryzen 1000 or later). Inference works on any x86-64 machine; performance is just lower without a GPU. Torch acceleration with BFloat16 for training and inference requires **AVX-512 BF16** (Intel Ice Lake-SP or newer) or **AMX** (Sapphire Rapids or newer, AMD Zen 4). The app detects your ISA at startup and selects the best available dtype automatically.

ARM is not currently tested or supported.

### C++ compiler (CPU inference only)
When no GPU is present, the app uses `torch.compile` (TorchInductor) to fuse and accelerate model inference. This requires a C++ compiler at runtime:

- **Windows:** **Build Tools for Visual Studio 2022** with the "Desktop development with C++" workload. MSVC is picked up automatically. MinGW is not supported by Inductor.
- **Linux:** **g++** — install with `sudo apt install build-essential` (Debian/Ubuntu) or `sudo dnf install gcc-c++` (Fedora).
- **MacOs:** **XCode Command Line Tools** - install with `xcode-select --install`. No Homebrew GCC needed — Apple's clang++ is what Inductor expects on MacOs.
*Without a compiler the app still runs — inference just won't be compiled.*

## Setup

Open the corresponding launcher - **launch_win.bat for Windows**, **launch_lin.sh for Linux and MacOs**. The launchers handle everything automatically: they install `uv`, create an isolated Python runtime, detect your GPU / CPU, install the correct PyTorch build, and verify the install before launching. You do not need to install anything manually except the C++ compiler above.

### 1. Get the repository

As this app in in its **beta phase**, you can get the full app bundle by:

- Cloning it:
```bash
git clone https://github.com/Programstor/Voice-Conversion-Ultimate
cd Voice-Conversion-Ultimate
```
- Downloading an archive and unpacking it manually: 
`Code(top right of the file list) -> Download ZIP` 

### 2. Download the pretrained models

Because of GitHub limitations, you will have to manually download the pretrained (.pth) models for training:

- **Recommended** are **the more modern TITAN pretrained models**, go to [TITAN Pretrains on HuggingFace](https://huggingface.co/blaise-tk/TITAN/tree/main/models/medium) and from the `40k/pretrained/` and `48k/pretrained/` folders download these:

```
G-f040k-TITAN-Medium.pth
D-f040k-TITAN-Medium.pth
G-f048k-TITAN-Medium.pth
D-f048k-TITAN-Medium.pth
```

- To get **the official pitch - aware (f0) RVC v2 pretrained models**, go to [RVC-WebUI on HuggingFace](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/pretrained_v2) and download these:

```
f0G40k.pth
f0D40k.pth
f0G48k.pth
f0D48k.pth
```

Place all of the pretrains in the `assets/pretrained/` folder inside the project.
These files must be present before you run **Model Training**!

### 3. Launch

**Windows:**
```
launch_win.bat
```

**Linux / MacOS:**
```bash
bash launch_lin.sh
```

On first run the launcher will set up the runtime (takes a few minutes). Subsequent launches skip setup unless your GPU / CPU changes or `requirements.txt` is updated.

You can override the detected backend at any time:
```bash
# Linux and MacOs
TORCH_BACKEND=cu126 bash launch_lin.sh

# Windows
set TORCH_BACKEND=cu126 && launch_win.bat
```

To force a full reinstall:
```bash
RECREATE=1 bash launch_lin.sh      # Linux and MacOs
set RECREATE=1 && launch_win.bat   # Windows
```

## App Guide

### Inference Tab

Convert an audio file to a target voice. Select a voice model (`.pth`) and an index file (`.index`) — the app auto-matches an index to the selected model by name. Browse for a source audio file (vocal track) and click **Infer Audio**. The result plays back in the built-in audio player.

Controls:

- **Transpose** — shift the output pitch up or down in semitones. Positive = higher (more feminine), negative = lower (more masculine). 0 = no shift.
- **Volume Scaling** — RMS envelope matching strength. 100% = output level tracks the source; 0% = use the model's own output level.
- **Voiceless Protection** — fraction of breath and consonant frames that bypass index retrieval and keep the original features. Lower values protect sibilants and breaths more aggressively to preserve the original's clarity.
- **Index Rate** — how strongly the retrieval index pulls the timbre toward the training data. Higher = more characteristic of the target voice, but can introduce artifacts on higher matching.
- **Resample to** — optionally resample the output to a standard rate before saving the audio. Set to 0 to keep the model's native sample rate.
- **Stitch Instruments** — when checked, instruments separated from the source are mixed back into the converted vocal output.
- **Keep Stereo** — applies *pseudo - stereo* to the converted output for more realism. Set its strength to your liking in the config.

### Training Tab

Train a new voice model from scratch using your own recordings.

**Model Options:**
- **Model name** — identifier for this run. All outputs (checkpoints, logs, index) are saved under `user/training/<name>/`. You must retype the same name later if you want to resume the training from a checkpoint or train an index!
- **Training folder** — folder containing your raw audio dataset. Defaults to `user/dataset/`.
- **Sample rate** — 40 kHz or 48 kHz. Must match the pretrained weights you intend to use. Choose before preprocessing; it cannot be changed mid-run.

**Training Options:**
- **Total Epochs** — number of full passes through the dataset. More epochs = more training, with diminishing returns and eventual overfitting past a point. 50–200 is typical depending on dataset size.
- **Save Frequency** — a checkpoint is written every this many epochs. Allows rolling back to an earlier state if quality degrades.
- **Batch Size** — number of audio chunks processed per gradient step. Increase if you have VRAM headroom; decrease if you get out-of-memory errors.

**Buttons:**
- **Preprocess** — slice, resample, and extract ContentVec features and F0 curves from the training folder. Must be run before training. Progress and any errors are shown in the status bar.
- **Train Model** — starts the training loop. The loss chart updates live during training.
- **Train Index** — builds a FAISS retrieval index from the extracted features after training. Use the resulting `.index` file alongside the `.pth` for best inference quality (automatically detected in the inference / realtime tabs).
- **Sequence** — runs Preprocess, Train Model, and Train Index sequentially and automatically resumes if the training is stopped.

Advanced training settings (AMP mode, CPU threads, TorchInductor compile, etc.) are controlled via `config.toml` — see [Config](#config).

### Realtime Tab

Live voice conversion through your microphone, routed to an output device. Select a voice model, index, microphone, and speaker, then click **Infer Voice** to start the stream. The button is a toggle — click again to stop.

Controls:

- **Transpose, Volume Scaling, Voiceless Protection, Index Rate** — same meaning as in Inference. All four update live while the stream is running without restarting it.
- **Block Length** — audio chunk converted per inference step, in milliseconds. This is the primary latency control. Lower = less delay but heavier CPU/GPU load and higher risk of dropout. 200–500 ms is a reasonable starting range; go lower only if your machine can keep up.
- **Noise Gate** — threshold below which a block is treated as silence and passed through without inference, saving computing during quiet moments.

## Config

Defaults for DSP, realtime, index, and training can be overridden without modifying source code by placing a `config.toml` in the project root. Only the keys you specify are changed; everything else keeps its default. *This is a demo of how the config should look like, not a recommended list of settings!*

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

Supported sections are `[dsp]`, `[realtime]`, `[index]`, `[train]`, and `[inference]`. Unknown keys are logged as warnings and ignored.

## To Do

- Features that will be implemented in the *Beta phase*:
`Pymsss Integration - for in-app vocal / instrumental seperation, which is now done with external apps like UVR or some online tools`
`ONNX Runtime Exports - for even faster CPU inference for users without access to advanced hardware or GPU in general`
`Extended Model Compatibility - support for inference and training of 32k models and older text - only (nof0) models`
`In-App Customization - make work with files easier and embedded into the app - configuring, themes and model exploring` 

- Features that are planned for the future, after *Release*:
`Diffusion Transform - state of the art inference, requiring no preprocess or model training, to produce a closer vocal and timbre match with the original speaker`
`BigVGAN v2 Vocoder - uses snake activations, which produce better results than the current VITS architecture's LeakyReLU activations. This is part of the DiT expansion.`

## License

This project is under the MIT license. For more details see the [LICENSE](LICENSE) file.

Made with love by <a href="https://github.com/Programstor" target="_blank">Programstor</a>

&#xa0;

<a href="#top">Back to top</a>