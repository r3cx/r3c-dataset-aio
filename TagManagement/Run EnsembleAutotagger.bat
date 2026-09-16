@echo off

:: Dataset Settings
set DATA_DIR="F:\StableDiffusion\Datasets\3 Pending\Noiretox\1_Extra"
set TRIGGER="@noiretox"
::

:: Gated models need a HF read token for download. Once downloaded, no credential is needed.
:: HF_TOKEN can be read from environment variables or set here.
::set "HF_TOKEN=hf_..."

set "VENV_DIR=%~dp0..\venv"

:: Check if python is accessible within the path
echo Detecting Virtual Environment...
dir "%VENV_DIR%\Scripts\Python.exe"
if %ERRORLEVEL% == 0 (
    :: Activate the virtual environment
    call "%VENV_DIR%\Scripts\activate.bat"

    :: Taggers: all four by default (one loaded at a time). To run a subset, set
    :: DEFAULT_TAGGERS in Autotaggers\EnsembleSWAutoTagger.py.

    :: Precision-speed trade-offs.
    :: - Default = --bf16 + --general_threshold=0.475 (lower threshold for more recall but more FPs)
    ::   - ~3x faster + ~half the VRAM; recall vs FP32 preserved)
    ::   - Recovers subtle tags caught by FP32 but around ~5% more minor false positives
    :: - FP32 (most precise, slowest): drop --bf16 and set the threshold back to 0.5.
    echo Starting EnsembleSWAutoTagger...
    python Autotaggers\EnsembleSWAutoTagger.py --data_dir %DATA_DIR% --general_threshold=0.475 --character_threshold=0.65 --num_data_loader_workers="6" --mode_append --use_gpu --bf16

    :: Start DatasetPreparer
    echo Starting DatasetPreparer...
    python Utility/DatasetPreparer.py --data_dir %DATA_DIR% --trigger_tag %TRIGGER%

    :: Deactivate the virtual environment
    call "%VENV_DIR%\Scripts\deactivate.bat"

    goto :end
)

echo Incomplete Setup Detected. Run 'Setup.bat' first!

:end
