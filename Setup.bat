@echo off

set "VENV_DIR=%~dp0%venv"

:: Check if python is accessible within the path
echo Detecting Virtual Environment...
dir "%VENV_DIR%\Scripts\Python.exe"
if %ERRORLEVEL% == 0 (
    goto :install
)
:: Else
echo Setting Up Virtual Environment...
python -m venv venv

:install
:: Activate the virtual environment
call .\venv\Scripts\activate.bat

:: Install requirements
echo Installing Package Requirements...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo Setup Complete!

::call BuildLLama.bat

:: Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat