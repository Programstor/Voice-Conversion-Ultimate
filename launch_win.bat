@echo off
setlocal EnableExtensions EnableDelayedExpansion
title Voice Conversion Ultimate - Windows Launcher and Setup

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

rem Remember whether the user explicitly selected a backend.
set "TORCH_BACKEND_SOURCE="
if not "%TORCH_BACKEND%"=="" set "TORCH_BACKEND_SOURCE=environment"

rem ----------------------------------------------------------------------
rem Logging helpers
rem ----------------------------------------------------------------------
goto :main

:log
if "%~1"=="" goto :log_blank
echo %*
echo %*>> "%LOG_FILE%"
exit /b 0

:log_blank
echo.
echo.>> "%LOG_FILE%"
exit /b 0

:logv
if "%~1"=="" goto :logv_blank
echo %*>> "%LOG_FILE%"
exit /b 0

:logv_blank
echo.>> "%LOG_FILE%"
exit /b 0

:main

rem ----------------------------------------------------------------------
rem 1. Decide whether setup needs to run
rem ----------------------------------------------------------------------
set "RUN_SETUP=0"

if not exist "%MARKER%" set "RUN_SETUP=1"
if not exist "%PYTHON_EXE%" set "RUN_SETUP=1"
if "!RECREATE!"=="1" set "RUN_SETUP=1"

rem Re-run setup if requirements.txt is newer than the setup marker.
if exist "%MARKER%" if exist "%REQ_FILE%" (
    for /f "delims=" %%I in ('xcopy /D /Y /L "%REQ_FILE%" "%MARKER%" 2^>nul ^| findstr /B /C:"1"') do (
        set "RUN_SETUP=1"
    )
)

rem ----------------------------------------------------------------------
rem 2. Backend mismatch check
rem ----------------------------------------------------------------------
if exist "%MARKER%" (
    set "_STORED_BACKEND="
    for /f "tokens=2 delims==" %%b in ('findstr /B "backend=" "%MARKER%" 2^>nul') do set "_STORED_BACKEND=%%b"

    if defined _STORED_BACKEND (
        set "_PHYSICAL_BACKEND=cpu"

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
            if !_DRIVER_MAJOR! GEQ 580 ( set "_PHYSICAL_BACKEND=cu130" ) else (
                if !_DRIVER_MAJOR! GEQ 525 ( set "_PHYSICAL_BACKEND=cu126" ) else (
                    set "_PHYSICAL_BACKEND=cpu"
                )
            )
        ) else (
            where rocm-smi >nul 2>&1
            if not errorlevel 1 set "_PHYSICAL_BACKEND=cpu_amd"
        )

        if defined TORCH_BACKEND (
            set "_DETECTED_BACKEND=%TORCH_BACKEND%"
        ) else (
            set "_DETECTED_BACKEND=!_PHYSICAL_BACKEND!"
        )

        if not "!_STORED_BACKEND!"=="!_DETECTED_BACKEND!" (
            if defined TORCH_BACKEND (
                echo   [Notice] TORCH_BACKEND='%TORCH_BACKEND%' override detected (runtime was built for '!_STORED_BACKEND!').
                echo   [Notice] TORCH_BACKEND='%TORCH_BACKEND%' override detected (runtime was built for '!_STORED_BACKEND!').>> "%LOG_FILE%"
                echo            Bypassing automatic reinstall for debugging session.
                echo            Bypassing automatic reinstall for debugging session.>> "%LOG_FILE%"
            ) else (
                echo   GPU environment changed: runtime was built for '!_STORED_BACKEND!', now '!_PHYSICAL_BACKEND!'.
                echo   GPU environment changed: runtime was built for '!_STORED_BACKEND!', now '!_PHYSICAL_BACKEND!'.>> "%LOG_FILE%"
                echo   Triggering reinstall...
                echo   Triggering reinstall...>> "%LOG_FILE%"
                set "RECREATE=1"
                set "RUN_SETUP=1"
            )
        )
    )
)

rem ----------------------------------------------------------------------
rem 3. Setup logic
rem ----------------------------------------------------------------------
if "!RUN_SETUP!"=="0" goto :skip_setup

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

call :log === Voice Conversion Ultimate - Windows Runtime Setup ===
call :log Log:     %LOG_FILE%
call :log Started: %DATE% %TIME%
call :log ""

call :log [1/5] Checking prerequisites...

if not exist "%REQ_FILE%" (
    call :log ERROR: requirements.txt not found: %REQ_FILE%
    goto :setup_fail
)

if not exist "%CHECK_SCRIPT%" (
    call :log ERROR: check_runtime.py not found: %CHECK_SCRIPT%
    goto :setup_fail
)

rem ----------------------------------------------------------------------
rem GPU/backend detection
rem ----------------------------------------------------------------------
if "!TORCH_BACKEND!"=="" (
    call :detect_backend
    set "TORCH_BACKEND=!_DETECTED_BACKEND!"

    if "!_GPU_KIND!"=="nvidia" (
        if defined DRIVER_MAJOR (
            if !DRIVER_MAJOR! GEQ 580 (
                echo Detected NVIDIA GPU - TORCH_BACKEND=!TORCH_BACKEND! [driver: !DRIVER_VERSION!]
                echo Detected NVIDIA GPU - TORCH_BACKEND=!TORCH_BACKEND! [driver: !DRIVER_VERSION!]>> "%LOG_FILE%"
            ) else (
                if !DRIVER_MAJOR! GEQ 525 (
                    echo Detected NVIDIA GPU - TORCH_BACKEND=!TORCH_BACKEND! [driver: !DRIVER_VERSION!]
                    echo Detected NVIDIA GPU - TORCH_BACKEND=!TORCH_BACKEND! [driver: !DRIVER_VERSION!]>> "%LOG_FILE%"
                ) else (
                    call :log WARNING: NVIDIA driver !DRIVER_VERSION! is too old for the selected CUDA builds - using CPU.
                    echo Detected NVIDIA GPU - TORCH_BACKEND=cpu [driver: !DRIVER_VERSION!]
                    echo Detected NVIDIA GPU - TORCH_BACKEND=cpu [driver: !DRIVER_VERSION!]>> "%LOG_FILE%"
                )
            )
        ) else (
            call :log WARNING: NVIDIA GPU detected, but its driver version could not be read.
            call :log Using the CPU backend.
        )
    ) else (
        if "!_GPU_KIND!"=="amd" (
            call :log Detected AMD GPU.
            call :log ROCm is not supported by this Windows launcher.
            call :log Falling back to the CPU build of PyTorch.
            call :log For AMD GPU acceleration, use the Linux launcher.
        ) else (
            call :log No supported GPU detected - using the CPU build of PyTorch.
            call :log Training will work but may be very slow. Inference is usable.
        )
    )

    echo(Override the backend with: set TORCH_BACKEND=^<tag^> ^&^& launch_win.bat
    echo(Override the backend with: set TORCH_BACKEND=^<tag^> ^&^& launch_win.bat>> "%LOG_FILE%"
) else (
    if "!TORCH_BACKEND_SOURCE!"=="environment" (
        echo TORCH_BACKEND=!TORCH_BACKEND! [from environment]
        echo TORCH_BACKEND=!TORCH_BACKEND! [from environment]>> "%LOG_FILE%"
    ) else (
        echo TORCH_BACKEND=!TORCH_BACKEND! [from existing runtime]
        echo TORCH_BACKEND=!TORCH_BACKEND! [from existing runtime]>> "%LOG_FILE%"
    )
)

call :log ""
call :log [2/5] Checking for uv...

set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"

where uv >nul 2>&1
if errorlevel 1 (
    call :log Installing uv...

    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"

    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"

    where uv >nul 2>&1
    if errorlevel 1 (
        call :log ERROR: uv could not be installed or is not on PATH.
        goto :setup_fail
    )
)

for /f "delims=" %%V in ('uv --version') do (
    echo %%V
    echo %%V>> "%LOG_FILE%"
)

call :log ""
call :log [3/5] Preparing Python %PYTHON_VERSION% runtime at %RUNTIME_DIR% ...

if "!RECREATE!"=="1" if exist "%RUNTIME_DIR%" (
    call :log RECREATE=1 - removing existing runtime.

    rem Safety check: refuse to delete the project root.
    if /I "!RUNTIME_DIR!"=="!ROOT_DIR!" (
        call :log ERROR: RUNTIME_DIR equals ROOT_DIR - aborting to prevent data loss.
        goto :setup_fail
    )

    if /I "!RUNTIME_DIR!"=="!ROOT_DIR!\" (
        call :log ERROR: RUNTIME_DIR equals ROOT_DIR - aborting to prevent data loss.
        goto :setup_fail
    )

    rmdir /s /q "!RUNTIME_DIR!" >nul 2>&1

    if exist "!RUNTIME_DIR!" (
        call :log ERROR: Cannot delete !RUNTIME_DIR!.
        call :log Close PyCharm, Python processes, or terminals locking files in runtime and retry.
        goto :setup_fail
    )
)

if exist "%PYTHON_EXE%" (
    call :log Reusing existing runtime.
) else (
    uv venv "%RUNTIME_DIR%" --python %PYTHON_VERSION% --seed

    if errorlevel 1 (
        call :log ERROR: Could not create runtime.
        goto :setup_fail
    )
)

if exist "%MARKER%" del "%MARKER%" >nul 2>&1

call :log ""
call :log [4/5] Installing packages from requirements.txt - backend: !TORCH_BACKEND! ...

uv pip install --python "%PYTHON_EXE%" --torch-backend "!TORCH_BACKEND!" -r "%REQ_FILE%"

if errorlevel 1 (
    call :log ERROR: Package installation failed.
    call :log Retrying with verbose logging...

    uv pip install --python "%PYTHON_EXE%" --torch-backend "!TORCH_BACKEND!" -r "%REQ_FILE%" -v >> "%LOG_FILE%" 2>&1

    goto :setup_fail
)

call :log ""
call :log [5/5] Verifying runtime...

set "CHECK_OUT=%TEMP%\vcu_check_%RANDOM%.txt"

"%PYTHON_EXE%" "%CHECK_SCRIPT%" > "!CHECK_OUT!" 2>&1
set "CHECK_RC=!errorlevel!"

type "!CHECK_OUT!"
type "!CHECK_OUT!" >> "%LOG_FILE%"
del "!CHECK_OUT!" >nul 2>&1

if not "!CHECK_RC!"=="0" (
    call :log ERROR: Verification failed. Fix the FAIL items above and rerun.
    goto :setup_fail
)

> "%MARKER%" echo backend=!TORCH_BACKEND!
>> "%MARKER%" echo date=%DATE% %TIME%

set "RECREATE="

call :log ""
call :log === Setup complete ===
call :log Finished: %DATE% %TIME%
call :log Runtime: %RUNTIME_DIR%
call :log ""

rem ----------------------------------------------------------------------
rem 4. Launch application
rem ----------------------------------------------------------------------
:skip_setup

rem Older setup markers might not contain a backend entry.
if "!TORCH_BACKEND!"=="" (
    call :detect_backend
    set "TORCH_BACKEND=!_DETECTED_BACKEND!"
)

if /i "!TORCH_BACKEND!"=="cpu" (
    set "CUDA_VISIBLE_DEVICES=-1"
    set "CUDA_HOME="
    set "CUDA_PATH="

    rem Initialize MSVC for CPU TorchInductor compilation.
    set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

    if exist "!VCVARS!" (
        call "!VCVARS!" >nul

        if errorlevel 1 (
            echo [VCU] WARNING: vcvars64.bat returned an error.
        )
    ) else (
        echo [VCU] WARNING: vcvars64.bat not found - TorchInductor compilation may fail.
    )

    if defined VCToolsInstallDir (
        set "CXX=!VCToolsInstallDir!bin\HostX64\x64\cl.exe"
        set "CC=!VCToolsInstallDir!bin\HostX64\x64\cl.exe"
        echo [VCU] Compiler: !VCToolsInstallDir!bin\HostX64\x64\cl.exe
    ) else (
        echo [VCU] WARNING: VCToolsInstallDir not set - MSVC initialization may have failed.
    )
)

echo Launching Voice Conversion Ultimate ...
"%PYTHON_EXE%" "%ROOT_DIR%\app.py" %*

if not "%VCU_NO_PAUSE%"=="1" pause
exit /b 0

rem ----------------------------------------------------------------------
rem Setup failure handler
rem ----------------------------------------------------------------------
:setup_fail
echo.
echo Setup failed - see "%LOG_FILE%".
if not "%VCU_NO_PAUSE%"=="1" pause
exit /b 1

rem ----------------------------------------------------------------------
rem Detect the backend supported by the current system.
rem Results:
rem   _DETECTED_BACKEND = cu130, cu126, or cpu
rem   _GPU_KIND         = nvidia, amd, or none
rem ----------------------------------------------------------------------
:detect_backend
set "_DETECTED_BACKEND=cpu"
set "_GPU_KIND=none"
set "DRIVER_VERSION="
set "DRIVER_MAJOR="

where nvidia-smi >nul 2>&1
if errorlevel 1 goto :detect_backend_amd

for /f "delims=" %%V in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do (
    if not defined DRIVER_VERSION set "DRIVER_VERSION=%%V"
)

if not defined DRIVER_VERSION goto :detect_backend_amd

set "_GPU_KIND=nvidia"

for /f "tokens=1 delims=." %%M in ("!DRIVER_VERSION!") do (
    set "DRIVER_MAJOR=%%M"
)

rem Reject an invalid major version before using a numeric comparison.
for /f "delims=0123456789" %%A in ("!DRIVER_MAJOR!") do (
    set "DRIVER_MAJOR="
)

if not defined DRIVER_MAJOR goto :eof

if !DRIVER_MAJOR! GEQ 580 (
    set "_DETECTED_BACKEND=cu130"
) else (
    if !DRIVER_MAJOR! GEQ 525 (
        set "_DETECTED_BACKEND=cu126"
    ) else (
        set "_DETECTED_BACKEND=cpu"
    )
)

goto :eof

:detect_backend_amd
where rocm-smi >nul 2>&1
if not errorlevel 1 set "_GPU_KIND=amd"

set "_DETECTED_BACKEND=cpu"
goto :eof