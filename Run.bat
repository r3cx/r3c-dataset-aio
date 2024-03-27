@echo off

set "VENV_DIR=%~dp0%venv"

:: Check if python is accessible within the path
echo Detecting Virtual Environment...
dir "%VENV_DIR%\Scripts\Python.exe"
if %ERRORLEVEL% == 0 (
    :: Activate the virtual environment
    call .\venv\Scripts\activate.bat

    :: Start AutoTagger
    echo Starting AutoTagger...
    python AutoTagger.py

    :: Deactivate the virtual environment
    call .\venv\Scripts\deactivate.bat

    goto :end
)

echo Incomplete Setup Detected. Run 'Setup.bat' first!

:end
