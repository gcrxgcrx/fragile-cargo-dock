# Ladder training queue + joined analysis.
#
# ---------------------------------------------------------------------------
# WHY A QUEUE AND NOT 8 SIMULTANEOUS LAUNCHES
#
# The first two attempts launched all 8 trainings at once. The second attempt
# (2026-09-20 ~05:00) deadlocked: 7 of 8 processes never finished spawning their
# SubprocVecEnv workers and sat at 0.0s CPU for minutes, while the 8th ran with
# 16 threads. The same machine also threw "Allocation error : not enough memory"
# from an unrelated PowerShell spawn at that moment.
#
# Measured at the time: 328 processes, 22.3 GB total working set (Defender,
# Edge, Steam, QQ, Sogou, Douyin...), 12.9 GB free physical, and only
# **6.7 GB free commit** against a 59 GB commit limit. Eight trainings x
# (1 parent + 6 env workers) is ~56 process spawns needing ~12 GB of commit,
# so the spawns stalled. The earlier successful 8-way launch had more headroom.
#
# Therefore: train in a queue with a small concurrency, an explicit commit
# guard, and hang detection. Hyperparameters are UNCHANGED (the pilot config,
# n_envs=6, 1.2M steps, seed 0) so the results stay comparable with the runs
# already in SESSION_STATE.md.
# ---------------------------------------------------------------------------
#
# Command pattern per candidate (verified against training/train_sb3_wrapper.py:
# --save-dir receives training_summary.json / model.zip / vecnormalize.pkl
# directly; log naming follows runs/env_007/ladder_<arm>_<cand>.log):
#
#   python -m training.train_sb3_wrapper \
#     --config configs/env007_terminal_rule_pilot.yaml \
#     --reward runs/env_007/prompt_ladder/<arm>/<cand>/reward_v1.py \
#     --run-name ladder_<arm>_<cand> \
#     --save-dir runs/env_007/ladder_train/<arm>_<cand> \
#     --total-timesteps 1200000 --eval-episodes 20 --seed 0
#
# The 8 (arm, cand) pairs are chosen for OUTCOME VARIANCE (2 four-pass L2
# candidates, 2 not; 4 L0 candidates), because correlating a zero-variance check
# with a zero-variance outcome is undefined. See SESSION_STATE.md section 5.

param(
  [int]$MaxParallel = 3,
  [int]$TotalTimesteps = 1200000,
  [int]$EvalEpisodes = 20,
  [int]$Seed = 0,
  [int]$AnalysisEpisodes = 60,
  [int]$SeedOffset = 30000,
  [double]$MinFreeCommitGB = 3.0,
  [int]$HangSeconds = 300,
  [int]$MaxAttempts = 2,
  [int]$PollSeconds = 20,
  [switch]$SkipAnalysis
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$Py = "D:\Code\python\research\llm_env_310\Scripts\python.exe"
$Config = "configs/env007_terminal_rule_pilot.yaml"
$TrainRoot = "runs/env_007/ladder_train"
$LogRoot = "runs/env_007"

$Pairs = @(
  @("L2", "cand_00"), @("L2", "cand_05"), @("L2", "cand_04"), @("L2", "cand_14"),
  @("L0", "cand_02"), @("L0", "cand_11"), @("L0", "cand_13"), @("L0", "cand_00")
)

function Get-FreeCommitGB {
  $os = Get-CimInstance Win32_OperatingSystem
  return [math]::Round($os.FreeVirtualMemory / 1MB, 2)
}

function Start-Candidate {
  param($Job)
  $arm = $Job.Arm; $cand = $Job.Cand; $name = "$arm`_$cand"
  $saveDir = Join-Path $Repo "$TrainRoot/$name"
  $log = Join-Path $Repo "$LogRoot/ladder_$name.log"
  $errLog = Join-Path $Repo "$LogRoot/ladder_$name.err.log"
  New-Item -ItemType Directory -Force -Path $saveDir | Out-Null

  $argList = @(
    "-m", "training.train_sb3_wrapper",
    "--config", $Config,
    "--reward", "runs/env_007/prompt_ladder/$arm/$cand/reward_v1.py",
    "--run-name", "ladder_$name",
    "--save-dir", "$TrainRoot/$name",
    "--total-timesteps", "$TotalTimesteps",
    "--eval-episodes", "$EvalEpisodes",
    "--seed", "$Seed"
  )
  # Cap BLAS/OpenMP threads: 8 procs x 16 default threads oversubscribes 32
  # logical cores and inflates the per-process thread arenas.
  $env:OMP_NUM_THREADS = "6"
  $env:MKL_NUM_THREADS = "6"
  $p = Start-Process -FilePath $Py -ArgumentList $argList -WorkingDirectory $Repo `
    -RedirectStandardOutput $log -RedirectStandardError $errLog -WindowStyle Hidden -PassThru
  Start-Sleep -Seconds 20   # stagger so the 6 SubprocVecEnv workers do not all spawn at once

  # The venv python.exe re-execs: the real trainer is this shim's python child.
  $realPid = $null
  for ($i = 0; $i -lt 15; $i++) {
    $child = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)" -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -eq "python.exe" } | Select-Object -First 1
    if ($child) { $realPid = $child.ProcessId; break }
    Start-Sleep -Seconds 2
  }
  $Job.ShimPid = $p.Id
  $Job.RealPid = $realPid
  $Job.Attempts++
  $Job.LastCpu = 0.0
  $Job.CpuChangedAt = Get-Date
  $Job.StartedAt = Get-Date
  Write-Host ("[start ] {0,-12} attempt {1} shim={2} real={3} commit_free={4}GB" -f `
    $name, $Job.Attempts, $p.Id, $realPid, (Get-FreeCommitGB))
  return $Job
}

$jobs = @()
foreach ($p in $Pairs) {
  $jobs += [pscustomobject]@{
    Arm = $p[0]; Cand = $p[1]; Name = "$($p[0])_$($p[1])"
    Attempts = 0; ShimPid = $null; RealPid = $null
    LastCpu = 0.0; CpuChangedAt = Get-Date; StartedAt = $null
    Done = $false; Failed = $false
  }
}

Write-Host "repo        : $Repo"
Write-Host "python      : $Py"
Write-Host "config      : $Config"
Write-Host "queue       : $($jobs.Count) candidates, max $MaxParallel parallel"
Write-Host "commit guard: start only if free commit > $MinFreeCommitGB GB (now $(Get-FreeCommitGB) GB)"
Write-Host ""

while ($true) {
  $running = @($jobs | Where-Object { -not $_.Done -and -not $_.Failed -and $_.RealPid })

  foreach ($j in $running) {
    $dir = Join-Path $Repo "$TrainRoot/$($j.Name)"
    if (Test-Path (Join-Path $dir "training_summary.json")) {
      $j.Done = $true
      Write-Host ("[done  ] {0,-12} summary written" -f $j.Name)
      continue
    }
    $pr = Get-Process -Id $j.RealPid -ErrorAction SilentlyContinue
    if (-not $pr) {
      Write-Host ("[died  ] {0,-12} process gone, no summary" -f $j.Name)
      $j.RealPid = $null
      if ($j.Attempts -ge $MaxAttempts) { $j.Failed = $true } else { Write-Host ("[retry ] {0}" -f $j.Name) }
      continue
    }
    if ($pr.CPU -gt $j.LastCpu + 1.0) {
      $j.LastCpu = $pr.CPU
      $j.CpuChangedAt = Get-Date
    } elseif (((Get-Date) - $j.CpuChangedAt).TotalSeconds -gt $HangSeconds) {
      Write-Host ("[hung  ] {0,-12} no CPU progress for ${HangSeconds}s -> kill+retry" -f $j.Name)
      Stop-Process -Id $j.RealPid -Force -ErrorAction SilentlyContinue
      if ($j.ShimPid) { Stop-Process -Id $j.ShimPid -Force -ErrorAction SilentlyContinue }
      $j.RealPid = $null
      # clean the partial monitor output of the hung attempt
      Remove-Item (Join-Path $dir "monitor") -Recurse -Force -ErrorAction SilentlyContinue
      if ($j.Attempts -ge $MaxAttempts) { $j.Failed = $true }
    }
  }

  $doneCount = @($jobs | Where-Object { $_.Done }).Count
  $failedCount = @($jobs | Where-Object { $_.Failed }).Count
  $pending = @($jobs | Where-Object { -not $_.Done -and -not $_.Failed -and -not $_.RealPid })
  $running = @($jobs | Where-Object { -not $_.Done -and -not $_.Failed -and $_.RealPid })
  Write-Host ("[{0}] done {1}/{2} failed {3} running {4} pending {5} commit_free {6}GB" -f `
    (Get-Date).ToString("HH:mm:ss"), $doneCount, $jobs.Count, $failedCount,
    $running.Count, $pending.Count, (Get-FreeCommitGB))

  if ($doneCount + $failedCount -eq $jobs.Count) { break }

  if ($pending.Count -gt 0 -and $running.Count -lt $MaxParallel) {
    if ((Get-FreeCommitGB) -lt $MinFreeCommitGB) {
      Write-Host ("        waiting for commit headroom (need > $MinFreeCommitGB GB)")
    } else {
      $j = $pending[0]
      Start-Candidate -Job $j | Out-Null
      continue   # re-evaluate immediately; Start-Candidate already slept 20s
    }
  }

  Start-Sleep -Seconds $PollSeconds
}

Write-Host ""
foreach ($j in $jobs) {
  $ok = Test-Path (Join-Path $Repo "$TrainRoot/$($j.Name)/training_summary.json")
  Write-Host ("[result] {0,-12} summary={1} attempts={2} failed={3}" -f $j.Name, $ok, $j.Attempts, $j.Failed)
}

if ($SkipAnalysis) { Write-Host "analysis skipped"; exit 0 }

Write-Host ""
Write-Host "=== running ladder_analysis.py ==="
$analysisLog = Join-Path $Repo "$LogRoot/ladder_analysis.log"
Push-Location $Repo
try {
  & $Py ladder_analysis.py --episodes $AnalysisEpisodes --seed-offset $SeedOffset 2>&1 |
    Tee-Object -FilePath $analysisLog
} finally {
  Pop-Location
}
Write-Host "analysis exit code: $LASTEXITCODE"
