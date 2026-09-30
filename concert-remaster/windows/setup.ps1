# Concert Remaster - one-time setup for Windows 10/11.
#
# Installs everything into the concert-remaster folder, nothing system-wide:
#   tools\uv          the installer that fetches Python and packages
#   tools\python      a private Python 3.12
#   .venv             the app's Python environment (AI libraries for your GPU)
#   tools\ffmpeg      audio/video decoding
#   models            every AI model, so the app works fully offline afterwards
#
# Run it again at any time to repair or update (it skips what is already there).

param(
    [ValidateSet("auto", "nvidia", "amd", "cpu")] [string] $Gpu = "auto",
    [switch] $SkipModels
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is 10x faster without the progress bar
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$App = Split-Path -Parent $PSScriptRoot
$Tools = Join-Path $App "tools"
New-Item -ItemType Directory -Force -Path $Tools | Out-Null
$Log = Join-Path $App "setup.log"
Start-Transcript -Path $Log -Append | Out-Null

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }
function Ok($text) { Write-Host "    $text" -ForegroundColor Green }
function Info($text) { Write-Host "    $text" }
function Download($url, $dest) {
    Info "downloading $url"
    for ($i = 1; $i -le 3; $i++) {
        try { Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing; return }
        catch { if ($i -eq 3) { throw } ; Start-Sleep -Seconds (5 * $i) }
    }
}
function Run($exe, [string[]] $arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$([IO.Path]::GetFileName($exe)) failed (exit code $LASTEXITCODE). See $Log" }
}

try {
    Write-Host ""
    Write-Host "  Concert Remaster setup" -ForegroundColor Magenta
    Write-Host "  Everything is installed inside: $App"
    Write-Host "  Needs about 20 GB of disk space and an internet connection (only this once)."

    # 1. uv: fetches Python and Python packages, fast and self-contained
    Step "Getting the installer (uv)"
    $Uv = Join-Path $Tools "uv\uv.exe"
    if (-not (Test-Path $Uv)) {
        $zip = Join-Path $env:TEMP "concert-remaster-uv.zip"
        Download "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" $zip
        Expand-Archive -Force $zip (Join-Path $Tools "uv")
        Remove-Item $zip
    }
    Ok "uv ready"

    # 2. Which graphics card?
    Step "Checking your graphics card"
    $cards = (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join "; "
    Info "found: $cards"
    if ($Gpu -eq "auto") {
        if ($cards -match "NVIDIA") { $Gpu = "nvidia" }
        elseif ($cards -match "AMD|Radeon") { $Gpu = "amd" }
        else { $Gpu = "cpu" }
    }
    $label = @{ nvidia = "NVIDIA (CUDA) - fastest"; amd = "AMD (DirectML) - experimental, some models fall back to the CPU"; cpu = "CPU only - slow for long shows" }[$Gpu]
    Ok "using: $label"
    Set-Content -Path (Join-Path $App "gpu.txt") -Value $Gpu

    # 3. Python 3.12 in a private environment
    Step "Setting up Python 3.12"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools "python"
    $Venv = Join-Path $App ".venv"
    $Py = Join-Path $Venv "Scripts\python.exe"
    if (-not (Test-Path $Py)) { Run $Uv @("venv", "--python", "3.12", $Venv) }
    Ok "Python ready"

    # 4. AI libraries for this GPU
    Step "Installing the AI libraries (several GB; this is the long part)"
    switch ($Gpu) {
        "nvidia" {
            # CUDA 12.6 build: current PyTorch, matches onnxruntime-gpu's CUDA 12, works with NVIDIA drivers 560+.
            Run $Uv @("pip", "install", "--python", $Py, "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu126")
            $extra = "gpu"; $runtime = "onnxruntime-gpu"
        }
        "amd" {
            # DirectML needs its own build of PyTorch (2.4.1).
            Run $Uv @("pip", "install", "--python", $Py, "torch-directml")
            $extra = "dml"; $runtime = "onnxruntime-directml"
        }
        default {
            Run $Uv @("pip", "install", "--python", $Py, "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu")
            $extra = "cpu"; $runtime = $null
        }
    }
    Run $Uv @("pip", "install", "--python", $Py, "-e", "$App[$extra,app]")
    if ($runtime) {
        # Whisper pulls in the CPU onnxruntime, which would hide the GPU build; keep only the GPU one.
        & $Uv pip uninstall --python $Py onnxruntime 2>$null
        Run $Uv @("pip", "install", "--python", $Py, "--reinstall-package", $runtime, $runtime)
    }
    Run $Uv @("pip", "install", "--python", $Py, "--upgrade", "yt-dlp[default]")   # YouTube changes often
    Ok "libraries installed"

    # 5. ffmpeg for mp3/mp4/m4a
    Step "Installing ffmpeg"
    $ffmpeg = Get-ChildItem -Path $Tools -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $ffmpeg) {
        $zip = Join-Path $env:TEMP "concert-remaster-ffmpeg.zip"
        Download "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip" $zip
        Expand-Archive -Force $zip (Join-Path $Tools "ffmpeg")
        Remove-Item $zip
    }
    Ok "ffmpeg ready"

    # 6. Check that the GPU is usable
    Step "Testing the processing device"
    Run $Py @("-m", "concert_remaster", "devices")

    # 7. All models, so the app never needs the internet again
    if (-not $SkipModels) {
        Step "Downloading every AI model (about 12 GB, only once)"
        Run $Py @("-m", "concert_remaster", "download-models", "--preset", "all", "--whisper", "large-v3")
    }

    # 8. Desktop shortcut
    Step "Creating a desktop shortcut"
    $start = Join-Path (Split-Path -Parent $App) "start.bat"
    if (-not (Test-Path $start)) { $start = Join-Path $App "windows\start.bat" }
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "Concert Remaster.lnk"))
    $link.TargetPath = $start
    $link.WorkingDirectory = Split-Path -Parent $start
    $link.IconLocation = "$env:SystemRoot\System32\imageres.dll,103"
    $link.Save()
    Ok "shortcut created"

    Write-Host ""
    Write-Host "  All done. Start the app with 'Concert Remaster' on your desktop (or start.bat)." -ForegroundColor Green
    Write-Host ""
}
catch {
    Write-Host ""
    Write-Host "  Setup stopped: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  Fix the problem (usually the internet connection or disk space) and run setup.bat again;"
    Write-Host "  it continues where it stopped. Details are in $Log"
    Write-Host ""
    Stop-Transcript | Out-Null
    exit 1
}
Stop-Transcript | Out-Null
