# Generate the v8 candidate family, then run wave 2 of the isolation training.
# Launched as ONE background job so nothing depends on the interactive turn.
#
# The API key is read at runtime from the reproduction script's line 19 (never printed,
# never written to a file).
$ErrorActionPreference = "Continue"
$Repo = "D:\Code\python\research\form_github\expert-reward-agent"
Set-Location $Repo
$Py = "D:\Code\python\research\llm_env_310\Scripts\python.exe"

# --- API key: read at runtime, never echoed --------------------------------
$keyLine = Get-Content "$Repo\runs\env_001\ablation_eureka_feedback_v4\reproduction\run_ablation_score_only_v4.ps1" |
  Where-Object { $_ -match 'DEEPSEEK_API_KEY' } | Select-Object -First 1
$key = [regex]::Match($keyLine, '"([^"]+)"').Groups[1].Value
if (-not $key) { Write-Output "FATAL: could not read the API key"; exit 2 }
$env:EUREKA_DEEPSEEK_API_KEY = $key
$env:DEEPSEEK_THINKING = "disabled"
Write-Output ("key loaded: length={0}" -f $key.Length)

# --- wait for the wave-1 trainings to finish -------------------------------
Write-Output "waiting for wave-1 trainings to finish..."
while ($true) {
  $running = Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.WorkingSet64 -gt 150MB }
  if (-not $running) { break }
  Start-Sleep -Seconds 30
}
Write-Output "wave-1 processes gone at $(Get-Date -Format HH:mm:ss)"

# --- generate the v8 family ------------------------------------------------
Write-Output "=== v8 generation ==="
& $Py pilot_generate_only.py `
  --config configs/env007_terminal_rule_pilot.yaml `
  --context runs/env_007/terminal_rule_pilot/seed_0/context `
  --prompt prompts/eureka_01_initial_reward_v8.md `
  --out runs/env_007/prompt_ladder_v8 --n 8 --temperature 0.7
Write-Output "generation exit=$LASTEXITCODE"

# --- wave 2 of the isolation training --------------------------------------
Write-Output "=== wave 2 training ==="
& "$Repo\runs\env_007\train_queue.ps1" -Spec runs/env_007/close_speed_spec_wave2.json `
  -MaxParallel 2 -StaggerSeconds 15 -LogDir runs/env_007
Write-Output "ALL DONE at $(Get-Date -Format HH:mm:ss)"
