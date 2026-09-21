# Clean restart: ONE v8 candidate, un-chained, so that nothing else can kill it.
# (Wave 3 died because I killed its wrapper job; a direct Start-Process does not care about
#  the harness job's lifetime.)
$ErrorActionPreference = "Continue"
$Repo = "D:\Code\python\research\form_github\expert-reward-agent"
Set-Location $Repo
$Py = "D:\Code\python\research\llm_env_310\Scripts\python.exe"
$env:OMP_NUM_THREADS = "6"
$env:MKL_NUM_THREADS = "6"

$d = "$Repo\runs\env_007\prompt_ladder_v8\cand_01\training"
New-Item -ItemType Directory -Force -Path $d | Out-Null
Remove-Item -Recurse -Force "$d\monitor" -ErrorAction SilentlyContinue

Write-Output "starting v8_cand_01 at $(Get-Date -Format HH:mm:ss)"
& $Py -m training.train_sb3_wrapper `
  --config configs/env007_terminal_rule_pilot.yaml `
  --reward runs/env_007/prompt_ladder_v8/cand_01/reward_v1.py `
  --run-name train_v8_cand_01 `
  --save-dir runs/env_007/prompt_ladder_v8/cand_01/training `
  --total-timesteps 1200000 --eval-episodes 20 --seed 0
Write-Output "train exit=$LASTEXITCODE at $(Get-Date -Format HH:mm:ss)"
