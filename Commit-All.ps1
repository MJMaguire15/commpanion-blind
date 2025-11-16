# Commit-All.ps1
[CmdletBinding()]
param(
  [string[]]$Repos = @("auto_avsr","av_hubert","face_alignment","face_detection"),
  [string]$Message = "A3 updates: lipread eval/tts/cropping tweaks",
  [switch]$UseLFS,
  [switch]$Standalone,                 # set remotes to standalone repos (repo-name + "-a3")
  [string]$GithubUser = "MJMaguire15"  # change if needed
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

# map upstream URLs for convenience
$Upstreams = @{
  "auto_avsr"      = "https://github.com/mpc001/auto_avsr.git"
  "av_hubert"      = "https://github.com/facebookresearch/av_hubert.git"
  "face_alignment" = "https://github.com/hhj1897/face_alignment.git"
  "face_detection" = "https://github.com/hhj1897/face_detection.git"
}

# Prefer venv python for freeze
$py = ".\.venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }

$info = @()

foreach ($r in $Repos) {
  if (!(Test-Path $r)) { Write-Warning "Skip missing repo: $r"; continue }
  Push-Location $r

  # --- ensure remotes point where we want ---
  if ($Standalone) {
    $target = "https://github.com/$GithubUser/$($r)-a3.git"
    $hasOrigin = git remote | Select-String "^origin$"
    if ($hasOrigin) {
      $cur = (& git remote get-url origin 2>$null).Trim()
      if ($cur -ne $target) {
        git remote rename origin fork 2>$null
        if (git remote | Select-String "^origin$") {
          git remote set-url origin $target
        } else {
          git remote add origin $target
        }
      }
    } else {
      git remote add origin $target
    }
    if ($Upstreams.ContainsKey($r) -and -not (git remote | Select-String "^upstream$")) {
      git remote add upstream $Upstreams[$r]
    }
  }

  # --- branch ---
  $br = (git branch --show-current 2>$null).Trim()
  if (-not $br) { $br = "a3-work-{0:yyyyMMdd_HHmm}" -f (Get-Date); git switch -c $br }

  # --- LFS (optional) ---
  if ($UseLFS) {
    git lfs install
    git lfs track "*.pth" "*.pt" "*.onnx" | Out-Null
    git add .gitattributes
  }

    # --- commit if dirty ---
  git add -A
  $status = git status --porcelain 2>$null
  if ($status) { git commit -m $Message } else { Write-Host "[$r] nothing to commit" }

  # --- push (always set upstream) ---
  if (git remote | Select-String "^origin$") {
    $pushOut = & git push -u origin $br 2>&1
    if ($LASTEXITCODE -ne 0) {
      $txt = "$pushOut"
      if ($txt -match "can not upload new objects to public fork") {
        Write-Warning "[$r] LFS blocked on public fork. Re-run with -Standalone (repo-name -a3)."
      } elseif ($txt -match "exceeds GitHub's file size limit|GH001 Large files detected") {
        Write-Host "[$r] Large files in history; migrating to LFS…"
        git lfs install
        git lfs track "*.pth" "*.pt" "*.onnx" | Out-Null
        git add .gitattributes
        git commit -m "Track model checkpoints with LFS" 2>$null
        git lfs migrate import --include="*.pth,*.pt,*.onnx" --include-ref=refs/heads/$br
        git push --force-with-lease origin $br
      } elseif ($txt -match "This repository is over its data quota") {
        Write-Warning "[$r] LFS quota hit. Consider pushing code-only (strip checkpoints) or enable LFS."
      } else {
        # try fetch/rebase then push once more
        git fetch origin --prune
        git pull --rebase origin $br
        git push
        if ($LASTEXITCODE -ne 0) {
          Write-Error "[$r] Push failed. Reason:`n$pushOut"
        }
      }
    }
  } else {
    Write-Warning "[$r] no 'origin' remote configured; skipping push."
  }


  $sha = (git rev-parse HEAD).Trim()
  $originUrl = (& git remote get-url origin 2>$null).Trim()
  Pop-Location
  $info += [pscustomobject]@{ Repo=$r; Branch=$br; SHA=$sha; Origin=$originUrl }
}

# Pin submodules (if any)
if (Test-Path ".gitmodules") {
  git add $Repos
  $rootStatus = git status --porcelain 2>$null
  if ($rootStatus) { git commit -m "Pin submodules to latest SHAs"; git push }
}

# REPRO + lock deps
"--- Revisions ($(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) ---" | Out-File -Encoding utf8 REPRO.md
$info | ForEach-Object { "{0,-14} {1} @ {2}" -f ($_.Repo+":"), $_.Origin, $_.SHA } | Out-File -Append -Encoding utf8 REPRO.md

& $py -m pip freeze | Out-File -Encoding ascii requirements.lock.txt
git add REPRO.md requirements.lock.txt
$rootStatus = git status --porcelain 2>$null
if ($rootStatus) { git commit -m "Record component SHAs + lock Python deps"; git push }
