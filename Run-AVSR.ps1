# Run-AVSR.ps1
# Usage: .\Run-AVSR.ps1 [-Speak] [-RootDir ".\roi_root"] [-List "list.csv"] [-Ckpt ".\auto_avsr\checkpoints\vsr_trlrs2lrs3vox2avsp_base.pth"]

[CmdletBinding()]
param(
  [switch]$Speak = $true,
  [string]$RootDir = ".\roi_root",
  [string]$List    = "list.csv",
  [string]$Ckpt    = ".\auto_avsr\checkpoints\vsr_trlrs2lrs3vox2avsp_base.pth"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

# Activate venv if available
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
  . ".\.venv\Scripts\Activate.ps1"
}

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Python venv not found at $py" }

# PYTHONPATH so eval.py can import its local packages
$env:PYTHONPATH = (Resolve-Path ".\auto_avsr").Path

# Sanity checks
if (-not (Test-Path $RootDir)) { throw "RootDir not found: $RootDir" }
if (-not (Test-Path $Ckpt))   { throw "Checkpoint not found: $Ckpt" }

# Build args
$ArgsList = @(
  ".\auto_avsr\eval.py",
  "--modality","video",
  "--root-dir",$RootDir,
  "--test-file",$List,
  "--pretrained-model-path",$Ckpt
)
if ($Speak) { $ArgsList += "--speak" }

# Run
& $py @ArgsList
