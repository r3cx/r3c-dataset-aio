@echo off

:: YOUR SETTINGS GO HERE
set DATA_DIR="F:\StableDiffusion\Datasets\3 Pending\Kaenuco\1_Extra"
set UPSCALE_DIR="F:\StableDiffusion\Datasets\3 Pending\Kaenuco\Upscaled"
set ONNX_MODEL="F:\StableDiffusion\Tools\r3c-dataset-aio\Upscalers\4xNomos8kDAT.onnx"
:: FINAL output size relative to the original (1 = same size, 2 = double, 4 = quadscaled)
set SCALE=2
:: Only upscale images smaller than this many megapixels (1 = 1024x1024). Leave blank to upscale all.
set TARGET_MP=1
:: Optional: native multiplier of the model (auto-measured via a probe inference, e.g. 4 for a 4x model). Leave blank.
set MODEL_SCALE=
:: --

:: Build optional arguments BEFORE the venv block. Windows expands %VAR% for a whole
:: parenthesized block up front, so EXTRA_ARGS must be finished here or it expands empty.
set "EXTRA_ARGS="
if defined TARGET_MP set "EXTRA_ARGS=%EXTRA_ARGS% --target_mp %TARGET_MP%"
if defined MODEL_SCALE set "EXTRA_ARGS=%EXTRA_ARGS% --model_scale %MODEL_SCALE%"

set "VENV_DIR=%~dp0..\venv"

:: Check if python is accessible within the path
echo Detecting Virtual Environment...
dir "%VENV_DIR%\Scripts\Python.exe"
if %ERRORLEVEL% == 0 (
    :: Activate the virtual environment
    call "%VENV_DIR%\Scripts\activate.bat"

    :: Start Upscaler
    echo Starting ImageUpscaler...
    python Utility\ImageUpscaler.py --onnx_model %ONNX_MODEL% --input_dir %DATA_DIR% --output_dir %UPSCALE_DIR% --scale %SCALE% %EXTRA_ARGS%

    :: Deactivate the virtual environment
    call "%VENV_DIR%\Scripts\deactivate.bat"

    goto :end
)

echo Incomplete Setup Detected. Run 'Setup.bat' first!

:end
