@echo off
REM Create the project conda environment INSIDE this folder (./env) and
REM install GPU-enabled PyTorch. Nothing is installed into the base env.
REM Run from the project root:  setup_env.bat

setlocal
set ENV_PREFIX=%~dp0env

if not exist "%ENV_PREFIX%" (
    echo Creating conda env at %ENV_PREFIX% ...
    call conda create --prefix "%ENV_PREFIX%" python=3.11 -y
) else (
    echo Env already exists at %ENV_PREFIX%; skipping create.
)

call "%ENV_PREFIX%\Scripts\activate.bat"

python -m pip install --upgrade pip
echo Installing PyTorch (CUDA 12.8, Blackwell sm_120 support) ...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
echo Installing remaining deps ...
pip install pygame tensorboard numpy

echo.
echo Done. Activate with:  %ENV_PREFIX%\Scripts\activate.bat
endlocal
