$ErrorActionPreference = "Continue"
Set-Location "D:\Code\python\research\form_github\expert-reward-agent"
$run = "runs\env_007\fragilecargo_eureka_v9fix"
while (-not (Test-Path "$run\seed_0\eureka_summary.md")) { Start-Sleep -Seconds 60 }
# run finished: score every trained candidate on one fresh block
$py = "D:\Code\python\research\llm_env_310\Scripts\python.exe"
$dirs = Get-ChildItem "$run\seed_0\gen_0*\cand_0*\training" -Directory | ForEach-Object { $_.FullName.Replace((Get-Location).Path+'\','').Replace('\','/') }
& $py eval_pool.py --episodes 60 --seed-offset 39000 --out "$run\eval_block39000.json" @dirs *> "$run\eval.log"
# and score the best candidate again on a second block for a stability check
$best = "$run\seed_0\best"
if (Test-Path "$best\model.zip") {
  & $py eval_pool.py --episodes 60 --seed-offset 40000 --out "$run\eval_best_block40000.json" "$run/seed_0/best" *> "$run\eval_best.log"
}
Write-Output "EUREKA-FIXEDSPEC DONE at $(Get-Date -Format HH:mm:ss)"
