@echo off
title Launching TensorBoard...

:: 1. Open the web browser to the local page in the background
echo Opening browser to http://localhost:6006/...
start "" "http://localhost:6006/"

:: 2. Wait 1 second to allow the server to fully initialize
timeout /t 1 /nobreak >nul

:: 3. Start TensorBoard (This keeps the command prompt window open)
echo Starting TensorBoard server...
tensorboard --logdir="F:/StableDiffusion/Datasets/Z Logs/Tensorboard"

pause
