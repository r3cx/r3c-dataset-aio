@echo off

:: Check if python is accessible within the path
IF NOT EXIST venv (
    echo Creating venv...
    python -m venv venv
)

:install
:: Activate the virtual environment
call .\venv\Scripts\activate.bat

:: Install requirements
echo Installing Requirements...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:: Start AutoTagger
echo Starting AutoTagger...
python AutoTagger.py

:: Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat