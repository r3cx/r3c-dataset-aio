@echo off

:: Tag Clean - Use the preparer to clean tags
set DATA_DIR="F:\StableDiffusion\Datasets\2 Baking\Uenomigi\1_Style"
set TRIGGER=""
:: --

set "VENV_DIR=%~dp0..\venv"

:: Check if python is accessible within the path
echo Detecting Virtual Environment...
dir "%VENV_DIR%\Scripts\Python.exe"
if %ERRORLEVEL% == 0 (
    :: Activate the virtual environment
    call "%VENV_DIR%\Scripts\activate.bat"

    :: Start DatasetPreparer
    echo Starting DatasetPreparer...
    python DatasetPreparer.py --data_dir %DATA_DIR% --trigger_tag %TRIGGER%

    :: Deactivate the virtual environment
    call "%VENV_DIR%\Scripts\deactivate.bat"

    goto :end
)

echo Incomplete Setup Detected. Run 'Setup.bat' first!

:end
