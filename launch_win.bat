@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Voice Conversion Ultimate - Windows Launcher & Setup

cd /d "%~dp0"
set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
set "VCU_HOME=%ROOT_DIR%"

set "RUNTIME_DIR=%ROOT_DIR%\runtime"
set "PYTHON_EXE=%RUNTIME_DIR%\Scripts\python.exe"
set "REQ_FILE=%ROOT_DIR%\requirements.txt"
set "CHECK_SCRIPT=%ROOT_DIR%\check_runtime.py"
set "MARKER=%RUNTIME_DIR%\.setup_complete"
set "LOG_DIR=%ROOT_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\setup_win.log"

if not defined PYTHON_VERSION set "PYTHON_VERSION=3.12"

rem ------------------------------------------------------------------------------
rem Logging helper — write to console AND log file without duplicating every line.
rem Usage: call :log <message>
rem ------------------------------------------------------------------------------
goto :main

:log
    echo %*
    echo %*>> "%LOG_FILE%"
    exit /b 0

:logv
    rem Log only to file (verbose / detail lines)
    echo %*>> "%LOG_FILE%"
    exit /b 0

:main

rem ------------------------------------------------------------------------------
rem 1. Decide whether setup needs to run
rem ------------------------------------------------------------------------------
set "RUN_SETUP=0"
if not exist "%MARKER%"    set "RUN_SETUP=1"
if not exist "%PYTHON_EXE%" set "RUN_SETUP=1"
if "%RECREATE%"=="1"       set "RUN_SETUP=1"

if exist "%MARKER%" if exist "%REQ_FILE%" (
    for /f "delims=" %%I in ('xcopy /D /Y /L "%REQ_FILE%" "%MARKER%" 2^>nul ^| findstr /B /C:"1"') do set "RUN_SETUP=1"
)

rem ------------------------------------------------------------------------------
rem 2. Backend mismatch check — force reinstall if the GPU environment changed
rem    Runs silently every launch; only speaks if the stored backend differs from
rem    what we would pick today.
rem ------------------------------------------------------------------------------
if exist "%MARKER%" (
    set "_STORED_BACKEND="
    for /f "tokens=2 delims==" %%b in ('findstr /B "backend=" "%MARKER%" 2^>nul') do set "_STORED_BACKEND=%%b"

    if defined _STORED_BACKEND (
        rem Detect current GPU silently
        set "_DETECTED_BACKEND=cpu"

        set "_DRIVER_MAJOR="
        where nvidia-smi >nul 2>&1
        if not errorlevel 1 (
            for /f "delims=" %%v in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do (
                if not defined _DV set "_DV=%%v"
            )
            if defined _DV (
                for /f "tokens=1 delims=." %%m in ("!_DV!") do set "_DRIVER_MAJOR=%%m"
                echo !_DRIVER_MAJOR!| findstr /r "^[0-9][0-9]*$" >nul || set "_DRIVER_MAJOR="
            )
        )

        if defined _DRIVER_MAJOR (
            if !_DRIVER_MAJOR! GEQ 580 ( set "_DETECTED_BACKEND=cu130" ) else (
                if !_DRIVER_MAJOR! GEQ 525 ( set "_DETECTED_BACKEND=cu126" ) else (
                    set "_DETECTED_BACKEND=cpu"
                )
            )
        ) else (
            rem No NVIDIA — check for AMD (ROCm not supported on Windows; CPU fallback)
            where rocm-smi >nul 2>&1
            if not errorlevel 1 set "_DETECTED_BACKEND=cpu_amd"
        )

        if not "!_STORED_BACKEND!"=="!_DETECTED_BACKEND!" (
            echo   GPU environment changed: runtime was built for '!_STORED_BACKEND!', now '!_DETECTED_BACKEND!'.
            echo   GPU environment changed: runtime was built for '!_STORED_BACKEND!', now '!_DETECTED_BACKEND!'.>> "%LOG_FILE%"
            echo   Triggering reinstall...
            echo   Triggering reinstall...>> "%LOG_FILE%"
            set "RECREATE=1"
            set "RUN_SETUP=1"
        )
    )
)

rem ------------------------------------------------------------------------------
rem 3. Setup Logic
rem ------------------------------------------------------------------------------
if "%RUN_SETUP%"=="1" (
    if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

    call :log === Voice Conversion Ultimate - Windows Runtime Setup ===
    call :log Log:     %LOG_FILE%
    call :log Started: %DATE% %TIME%
    call :log.

    call :log [1/5] Checking prerequisites...
    if not exist "%REQ_FILE%" (
        call :log ERROR: requirements.txt not found: %REQ_FILE%
        goto :setup_fail
    )
    if not exist "%CHECK_SCRIPT%" (
        call :log ERROR: check_runtime.py not found: %CHECK_SCRIPT%
        goto :setup_fail
    )

    rem ── GPU detection ──────────────────────────────────────────────────────────
    if not defined TORCH_BACKEND (
        set "DRIVER_VERSION="
        set "DRIVER_MAJOR="

        where nvidia-smi >nul 2>&1
        if not errorlevel 1 (
            for /f "delims=" %%v in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do (
                if not defined DRIVER_VERSION set "DRIVER_VERSION=%%v"
            )
            if defined DRIVER_VERSION (
                for /f "tokens=1 delims=." %%m in ("!DRIVER_VERSION!") do set "DRIVER_MAJOR=%%m"
                echo !DRIVER_MAJOR!| findstr /r "^[0-9][0-9]*$" >nul || set "DRIVER_MAJOR="
            )
        )

        if defined DRIVER_MAJOR (
            rem NVIDIA GPU found — pick CUDA wheel by driver version
            if !DRIVER_MAJOR! GEQ 580 ( set "TORCH_BACKEND=cu130" ) else (
                if !DRIVER_MAJOR! GEQ 525 ( set "TORCH_BACKEND=cu126" ) else (
                    set "TORCH_BACKEND=cpu"
                    call :log   WARNING: NVIDIA driver !DRIVER_VERSION! is too old for CUDA - using CPU.
                )
            )
            call :log   Detected NVIDIA GPU -- TORCH_BACKEND=!TORCH_BACKEND! ^(driver: !DRIVER_VERSION!^)
        ) else (
            rem No NVIDIA — check for AMD
            where rocm-smi >nul 2>&1
            if not errorlevel 1 (
                rem AMD GPU present but ROCm is not supported on Windows.
                rem PyTorch has no Windows ROCm wheels; falling back to CPU.
                set "TORCH_BACKEND=cpu"
                call :log   Detected AMD GPU -- ROCm is not supported on Windows.
                call :log   Falling back to CPU build of PyTorch.
                call :log   For GPU acceleration on AMD hardware please use the Linux launcher.
            ) else (
                set "TORCH_BACKEND=cpu"
                call :log   No supported GPU found - using CPU build of PyTorch.
                call :log   Training will work but is very slow. Inference is usable.
            )
        )
        call :log   Override any time with:  set TORCH_BACKEND=^<tag^) ^&^& run_windows.bat
    ) else (
        call :log   TORCH_BACKEND=%TORCH_BACKEND% ^(from environment^)
    )

    call :log.
    call :log [2/5] Checking for uv...
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
    where uv >nul 2>&1
    if errorlevel 1 (
        call :log   Installing uv...
        powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
        where uv >nul 2>&1
        if errorlevel 1 (
            call :log ERROR: uv could not be installed or is not on PATH.
            goto :setup_fail
        )
    )
    for /f "delims=" %%v in ('uv --version') do call :log   %%v

    call :log.
    call :log [3/5] Preparing Python %PYTHON_VERSION% runtime at %RUNTIME_DIR% ...
    if "%RECREATE%"=="1" if exist "%RUNTIME_DIR%" (
        call :log   RECREATE=1 - removing existing runtime.
        rmdir /s /q "%RUNTIME_DIR%"
    )

    if exist "%PYTHON_EXE%" (
        call :log   Reusing existing runtime.
    ) else (
        uv venv "%RUNTIME_DIR%" --python %PYTHON_VERSION% --seed
        if errorlevel 1 (
            call :log ERROR: Could not create runtime.
            goto :setup_fail
        )
    )
    if exist "%MARKER%" del "%MARKER%"

    call :log.
    call :log [4/5] Installing packages from requirements.txt ^(PyTorch backend: !TORCH_BACKEND!^) ...
    uv pip install --python "%PYTHON_EXE%" --torch-backend !TORCH_BACKEND! -r "%REQ_FILE%"
    if errorlevel 1 (
        call :log ERROR: Package installation failed. Run 'uv self update' if error mentions --torch-backend.
        uv pip install --python "%PYTHON_EXE%" --torch-backend !TORCH_BACKEND! -r "%REQ_FILE%" -v >> "%LOG_FILE%" 2>&1
        goto :setup_fail
    )

    call :log.
    call :log [5/5] Verifying runtime...
    set "CHECK_OUT=%TEMP%\rvc_check_%RANDOM%.txt"
    "%PYTHON_EXE%" "%CHECK_SCRIPT%" > "!CHECK_OUT!" 2>&1
    set "CHECK_RC=!errorlevel!"
    type "!CHECK_OUT!"
    type "!CHECK_OUT!" >> "%LOG_FILE%"
    del "!CHECK_OUT!" >nul 2>&1
    if not "!CHECK_RC!"=="0" (
        call :log ERROR: Verification failed. Fix [FAIL] items above and rerun.
        goto :setup_fail
    )

    > "%MARKER%" echo backend=!TORCH_BACKEND!
    >> "%MARKER%" echo date=%DATE% %TIME%

    call :log.
    call :log === Setup complete ===
    call :log Finished: %DATE% %TIME%
    call :log Runtime:  %RUNTIME_DIR%
    call :log.
)

rem ------------------------------------------------------------------------------
rem 4. Launch Application
rem ------------------------------------------------------------------------------
echo Launching Voice Conversion Ultimate ...
"%PYTHON_EXE%" "%ROOT_DIR%\app.py" %*

if not "%VCU_NO_PAUSE%"=="1" pause
exit /b 0

:setup_fail
echo.
echo Setup failed - see %LOG_FILE%
if not "%VCU_NO_PAUSE%"=="1" pause
exit /b 1
