# Generate the v8 candidate family (retry: the previous attempt used a placeholder key).
# The key is read from outside the repo so it is never written into it.
$ErrorActionPreference = "Continue"
$Repo = "D:\Code\python\research\form_github\expert-reward-agent"
Set-Location $Repo
$Py = "D:\Code\python\research\llm_env_310\Scripts\python.exe"

$key = ([System.IO.File]::ReadAllText("D:\Code\python\research\DSapi.txt")).Trim()
if (-not $key.StartsWith("sk-")) { Write-Output "FATAL: key file does not contain a key"; exit 2 }
$env:EUREKA_DEEPSEEK_API_KEY = $key
$env:DEEPSEEK_API_KEY = $key
$env:DEEPSEEK_THINKING = "disabled"
Write-Output ("key loaded: len={0}" -f $key.Length)

Write-Output "=== v8 generation (N=8) ==="
& $Py pilot_generate_only.py `
  --config configs/env007_terminal_rule_pilot.yaml `
  --context runs/env_007/terminal_rule_pilot/seed_0/context `
  --prompt prompts/eureka_01_initial_reward_v8.md `
  --out runs/env_007/prompt_ladder_v8 --n 8 --temperature 0.7
Write-Output "generation exit=$LASTEXITCODE"

Write-Output "=== mechanical form check (lint only, not a selector) ==="
$rewards = 0..7 | ForEach-Object { "runs/env_007/prompt_ladder_v8/cand_0$_/reward_v1.py" } |
  Where-Object { Test-Path $_ }
if ($rewards.Count -gt 0) {
  & $Py check_settled_stream.py @rewards
  & $Py check_v8_rules.py @rewards
} else {
  Write-Output "no valid candidates were written"
}
Write-Output "ALL DONE at $(Get-Date -Format HH:mm:ss)"
