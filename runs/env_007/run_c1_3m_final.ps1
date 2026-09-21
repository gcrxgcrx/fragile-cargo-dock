# Final experiment of the session: does a 1.2 M run predict a seed's class at 3 M?
# Two runs of C1 (= control_v1 + one closing-speed term), 3 M steps, seeds 0 and 2.
# Seed 0 scored 45/60 at 1.2 M; seed 2 scored 0/60 with dock 0.02 (never approached).
$ErrorActionPreference = "Continue"
$Repo = "D:\Code\python\research\form_github\expert-reward-agent"
Set-Location $Repo
$py = "D:\Code\python\research\llm_env_310\Scripts\python.exe"
$env:OMP_NUM_THREADS = "6"; $env:MKL_NUM_THREADS = "6"

$jobs = @(
  @{ seed = 0; save = "runs/env_007/close_speed_test/C1_closing_speed/seed0_3M"; name = "cs_C1_s0_3M" },
  @{ seed = 2; save = "runs/env_007/close_speed_test/C1_closing_speed/seed2_3M"; name = "cs_C1_s2_3M" }
)
$procs = @()
foreach ($j in $jobs) {
  $log = "$Repo\runs\env_007\train_$($j.name).log"
  $err = "$Repo\runs\env_007\train_$($j.name).err.log"
  New-Item -ItemType Directory -Force -Path "$Repo\$($j.save)" | Out-Null
  $a = @("-m","training.train_sb3_wrapper","--config","configs/env007_terminal_rule_pilot.yaml",
    "--reward","runs/env_007/close_speed_test/C1_closing_speed/reward_v1.py",
    "--run-name","train_$($j.name)","--save-dir",$j.save,
    "--total-timesteps","3000000","--eval-episodes","20","--seed","$($j.seed)")
  $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $Repo -WindowStyle Hidden -PassThru `
       -RedirectStandardOutput $log -RedirectStandardError $err
  Write-Output ("launched {0} (seed {1}) pid={2}" -f $j.name, $j.seed, $p.Id)
  $procs += $p
  Start-Sleep -Seconds 20
}
Write-Output "both launched at $(Get-Date -Format HH:mm:ss)"
