$ErrorActionPreference = "Continue"
$Repo = "D:\Code\python\research\form_github\expert-reward-agent"
Set-Location $Repo
while ($true) {
  $running = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.WorkingSet64 -gt 150MB }
  if (-not $running) { break }
  Start-Sleep -Seconds 30
}
Write-Output "wave 3 gone at $(Get-Date -Format HH:mm:ss); starting wave 4"
& "$Repo\runs\env_007\train_queue.ps1" -Spec runs/env_007/v8_spec_wave4.json -MaxParallel 2 -StaggerSeconds 15 -LogDir runs/env_007
Write-Output "WAVE4 DONE at $(Get-Date -Format HH:mm:ss)"
