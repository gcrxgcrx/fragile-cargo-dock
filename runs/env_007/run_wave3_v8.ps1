# Wait for wave 2 to finish, then run wave 3 (the three clean v8 candidates).
# Guarded so the machine is never asked to run more than two trainings at once.
$ErrorActionPreference = "Continue"
$Repo = "D:\Code\python\research\form_github\expert-reward-agent"
Set-Location $Repo

Write-Output "waiting for wave 2 (started $(Get-Date -Format HH:mm:ss))"
while ($true) {
  $running = Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.WorkingSet64 -gt 150MB }
  if (-not $running) { break }
  Start-Sleep -Seconds 30
}
Write-Output "wave 2 done at $(Get-Date -Format HH:mm:ss)"

& "$Repo\runs\env_007\train_queue.ps1" -Spec runs/env_007/v8_spec_wave3.json `
  -MaxParallel 2 -StaggerSeconds 15 -LogDir runs/env_007
Write-Output "WAVE3 DONE at $(Get-Date -Format HH:mm:ss)"
