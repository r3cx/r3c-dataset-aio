@echo off

:: YOUR SETTINGS GO HERE
set DATA_DIR="F:\StableDiffusion\Datasets\2 Baking\Serin199 (2)\Extra"
set UPSCALE_DIR="F:\StableDiffusion\Datasets\2 Baking\Serin199 (2)\Upscaled"
set ONNX_MODEL="%~dp0..\Upscalers\4xNomos8kDAT.onnx"
set SCALE=1.5
:: --

set "VENV_DIR=%~dp0..\venv"

:: Check if python is accessible within the path
echo Detecting Virtual Environment...
dir "%VENV_DIR%\Scripts\Python.exe"
if %ERRORLEVEL% == 0 (
    :: Activate the virtual environment
    call "%VENV_DIR%\Scripts\activate.bat"

    :: Start Upscaler
    echo Starting ImageUpscaler...
    python ..\Utilities\Image\ImageUpscaler.py --onnx_model %ONNX_MODEL% --input_dir %DATA_DIR% --output_dir %UPSCALE_DIR% --scale %SCALE%

    :: Deactivate the virtual environment
    call "%VENV_DIR%\Scripts\deactivate.bat"

    goto :end
)

echo Incomplete Setup Detected. Run 'Setup.bat' first!

:end
