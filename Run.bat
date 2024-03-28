@echo off

:: YOUR SETTINGS GO HERE
set DATA_DIR="F:\StableDiffusion\Datasets\Z Tools\Test"
:: --


set "VENV_DIR=%~dp0%venv"

:: Check if python is accessible within the path
echo Detecting Virtual Environment...
dir "%VENV_DIR%\Scripts\Python.exe"
if %ERRORLEVEL% == 0 (
    :: Activate the virtual environment
    call .\venv\Scripts\activate.bat

    :: Start AutoTagger
    echo Starting AutoTagger...
    python AutoTagger.py --general_threshold=0.25 --character_threshold=1 --num_data_loader_workers="2" --frequency_tags --data_dir %DATA_DIR%

    :: Start DatasetPreparer
    echo Starting DatasetPreparer...
    python DatasetPreparer.py --data_dir %DATA_DIR%

    :: Deactivate the virtual environment
    call .\venv\Scripts\deactivate.bat

    goto :end
)

echo Incomplete Setup Detected. Run 'Setup.bat' first!

:end
