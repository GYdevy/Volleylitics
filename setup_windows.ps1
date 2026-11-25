param(
    [ValidateSet("auto", "amd", "nvidia", "cpu")]
    [string]$Backend = "auto"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "Volleylitics Windows setup" -ForegroundColor Green

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw "Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/windows/ and enable 'Add Python to PATH'."
}

& $python.Source -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.10 or newer is required."
}

$venvPython = Join-Path $PSScriptRoot ".venv-windows\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating the Windows virtual environment..."
    & $python.Source -m venv .venv-windows
}

Write-Host "Installing Python packages. This can take several minutes..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }

$detectedArch = $null
$hipPath = $env:HIP_PATH
if (-not $hipPath -and (Test-Path -LiteralPath "C:\Program Files\AMD\ROCm")) {
    $latestRocm = Get-ChildItem "C:\Program Files\AMD\ROCm" -Directory | Sort-Object Name -Descending | Select-Object -First 1
    if ($latestRocm) { $hipPath = $latestRocm.FullName }
}
if ($hipPath) {
    $archTool = Join-Path $hipPath "bin\amdgpu-arch.exe"
    if (Test-Path -LiteralPath $archTool) {
        $detectedArch = (& $archTool 2>$null | Select-Object -First 1).Trim()
    }
}

if ($Backend -eq "auto") {
    if ($detectedArch -match '^gfx\d+$') { $Backend = "amd" }
    elseif (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $Backend = "nvidia" }
    else { $Backend = "cpu" }
}

switch ($Backend) {
    "amd" {
        if ($detectedArch -notmatch '^gfx\d+$') {
            throw "AMD ROCm was selected, but amdgpu-arch.exe could not identify the GPU architecture."
        }
        Write-Host "Installing AMD ROCm PyTorch for $detectedArch..." -ForegroundColor Cyan
        # TheRock device wheels are required for RDNA3 cards such as the RX 7800 XT.
        # Keep these exact versions paired; mixing the CPU PyPI wheel breaks ROCm.
        & $venvPython -m pip install --upgrade --force-reinstall `
            --index-url "https://stable.repo.amd.com/rocm/whl-next/" `
            --extra-index-url "https://pypi.org/simple" `
            "torch[device-$detectedArch]==2.13.0+rocm10.0.0" `
            "torchvision[device-$detectedArch]==0.28.0+rocm10.0.0"
        if ($LASTEXITCODE -ne 0) { throw "AMD ROCm PyTorch installation failed." }
    }
    "nvidia" {
        Write-Host "Installing NVIDIA CUDA 13.0 PyTorch..." -ForegroundColor Cyan
        & $venvPython -m pip install --upgrade --force-reinstall torch torchvision `
            --index-url "https://download.pytorch.org/whl/cu130"
        if ($LASTEXITCODE -ne 0) { throw "NVIDIA CUDA PyTorch installation failed." }
    }
    "cpu" {
        Write-Host "No supported GPU runtime detected; installing CPU PyTorch." -ForegroundColor Yellow
        & $venvPython -m pip install --upgrade torch torchvision
        if ($LASTEXITCODE -ne 0) { throw "CPU PyTorch installation failed." }
    }
}

& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Project dependency installation failed." }

Write-Host "Verifying the selected compute backend..."
& $venvPython -c "import torch; expected='$Backend'; hip=getattr(torch.version,'hip',None); cuda=getattr(torch.version,'cuda',None); print('PyTorch:',torch.__version__); print('HIP:',hip); print('CUDA:',cuda); print('GPU available:',torch.cuda.is_available()); print('Device:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); x=torch.arange(16,device='cuda' if expected in ('amd','nvidia') else 'cpu',dtype=torch.float32); value=(x*x).sum().item(); torch.cuda.synchronize() if x.is_cuda else None; assert value == 1240.0; assert expected != 'amd' or hip; assert expected != 'nvidia' or cuda"
if ($LASTEXITCODE -ne 0) {
    throw "The $Backend backend installed but failed its GPU calculation. If the AMD or NVIDIA driver was just updated, restart Windows and run setup_windows.ps1 again."
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "FFmpeg is not installed or is not on PATH."
    Write-Host "Install it from a new PowerShell window with: winget install --id Gyan.FFmpeg -e" -ForegroundColor Yellow
} else {
    Write-Host "FFmpeg found." -ForegroundColor Green
}

$models = @(
    "models\best_resnet_whistle_mix.pth",
    "models\rally_model_best.pth",
    "models\temp_best_ball.pt"
)
$missingModels = $models | Where-Object { -not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $_)) }
if ($missingModels.Count -gt 0) {
    Write-Warning "The following model files are still required:"
    $missingModels | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
    Write-Host "See RUNNING.md for model placement details."
} else {
    Write-Host "All model files found." -ForegroundColor Green
}

Write-Host ""
Write-Host "Setup complete. Double-click run_windows.bat to open the app." -ForegroundColor Green
