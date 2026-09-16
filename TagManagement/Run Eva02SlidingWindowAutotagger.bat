@echo off

:: YOUR SETTINGS GO HERE
set DATA_DIR=""
set TRIGGER=""
::set QUALITY=""   ::Fill in your own tags and add --quality_tags QUALITY to the call to DatasetPreparer.py to use your own set of tags
::set UNDESIRED="" ::Fill in your own tags and add --undesired_tags UNDESIRED to the call to DatasetPreparer.py to use your own set of tags
:: --
:: Only needed for the FIRST download of a gated model (see the Ensemble bat for the gated v4 taggers); a fully-populated Models/ needs no credential.
::set "HF_TOKEN=hf_..."
:: --

set "VENV_DIR=%~dp0..\venv"

:: Check if python is accessible within the path
echo Detecting Virtual Environment...
dir "%VENV_DIR%\Scripts\Python.exe"
if %ERRORLEVEL% == 0 (
    :: Activate the virtual environment
    call "%VENV_DIR%\Scripts\activate.bat"

    :: Start AutoTagger
    echo Starting Eva02SlidingWindowAutotagger...
    python Autotaggers\Eva02SlidingWindowAutotagger.py --general_threshold=0.65 --character_threshold=1 --num_data_loader_workers="6" --mode_append --data_dir %DATA_DIR% --use_gpu
    REM --debug_print

    :: Start DatasetPreparer
    echo Starting DatasetPreparer...
    python Utility/DatasetPreparer.py --data_dir %DATA_DIR% --trigger_tag %TRIGGER%

    :: Deactivate the virtual environment
    call "%VENV_DIR%\Scripts\deactivate.bat"

    goto :end
)

echo Incomplete Setup Detected. Run 'Setup.bat' first!

:end
