$ErrorActionPreference = "Stop"

Write-Host "Building NeuroSeq Desktop..." -ForegroundColor Cyan

python -m pip install --upgrade pyinstaller pillow

python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --name "NeuroSeq" `
    --icon "GUI\resources\NeuroSeq_logo.ico" `
    --add-data "GUI\resources;GUI\resources" `
    --collect-all matplotlib `
    --collect-all sklearn `
    --collect-all networkx `
    --collect-all pandas `
    --collect-all numpy `
    GUI\launcher.py

Write-Host ""
Write-Host "Build complete:" -ForegroundColor Green
Write-Host "dist\NeuroSeq\NeuroSeq.exe"
