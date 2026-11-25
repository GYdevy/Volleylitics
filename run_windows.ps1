$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv-windows\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    & (Join-Path $PSScriptRoot "setup_windows.ps1")
}

& (Join-Path $PSScriptRoot "windows_gui.ps1")
