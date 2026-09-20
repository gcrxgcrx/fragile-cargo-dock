# `ladder_train/` — how these 8 trainings are launched, and the commit-limit trap

## What is here

Eight 1.2M-step trainings, one per `(arm, cand)` pair, chosen so the *outcome* has
variance (correlating a zero-variance check with a zero-variance outcome is undefined —
DeepSeek's correction 10):

| arm | candidates | why |
|---|---|---|
| `L2` (oracle scaffold) | `cand_00`, `cand_05`, `cand_04`, `cand_14` | 2 four-pass, 2 not |
| `L0` (interface only) | `cand_02`, `cand_11`, `cand_13`, `cand_00` | `cand_02` is the single L0 candidate that writes an event |

Rewards live in `runs/env_007/prompt_ladder/<arm>/<cand>/reward_v1.py`.
Outputs land in `runs/env_007/ladder_train/<arm>_<cand>/`
(`training_summary.json`, `model.zip`, `vecnormalize.pkl`, `monitor/`), and stdout in
`runs/env_007/ladder_<arm>_<cand>.log`.

## Command pattern

Hyperparameters are the established ones (`configs/env007_terminal_rule_pilot.yaml`,
`n_envs=6`, clip 20, 1.2M steps, seed 0) so these runs stay comparable with everything in
`SESSION_STATE.md`.

```powershell
python -m training.train_sb3_wrapper `
  --config configs/env007_terminal_rule_pilot.yaml `
  --reward runs/env_007/prompt_ladder/<arm>/<cand>/reward_v1.py `
  --run-name ladder_<arm>_<cand> `
  --save-dir runs/env_007/ladder_train/<arm>_<cand> `
  --total-timesteps 1200000 --eval-episodes 20 --seed 0
```

`--save-dir` receives the summaries **directly** (no extra `training/` level).
`ladder_analysis.py` runs afterwards and joins structural checks with fresh-seed success.

Run it via:

```powershell
& runs/env_007/ladder_train/launch_and_analyze.ps1 -MaxParallel 4
```

## The trap: free **commit**, not free RAM

Launching all 8 at once killed the second attempt (2026-09-20 ~05:00). Only one candidate
progressed; the other seven sat at ~0 CPU for minutes with their `SubprocVecEnv` workers
only partly spawned. Their stderr logs hold the proof:

```
MemoryError                                     # importing torch in a worker
ImportError: DLL load failed while importing bit_generator: 页面文件太小，无法完成操作
OSError: [WinError 1455] 页面文件太小，无法完成操作   # ERROR_COMMITMENT_LIMIT, c10.dll
OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败  # shm.dll
```

The parent trainers then blocked waiting on workers that never came up — which looks
exactly like a hang, not like an error, unless you read the `.err.log`.

Measured at the time: **6.7 GB free commit** against a 59 GB limit, with 22.3 GB already
held by unrelated desktop software (Defender, Edge, Steam, QQ, Sogou, Douyin — 328
processes), and free *physical* memory still looked healthy at 12.9 GB. RAM being free is
therefore not sufficient evidence that a launch is safe: 8 trainers x (1 parent + 6
workers) is ~56 process spawns, each importing torch.

The same pressure surfaced independently as `Allocation error : not enough memory` from an
unrelated PowerShell spawn.

### The fix, as implemented in `launch_and_analyze.ps1`

* queue the candidates instead of launching all 8 — `-MaxParallel 4`;
* start only when free commit is above `-MinFreeCommitGB 3.0`;
* stagger starts by 20 s so the 6 workers of each candidate do not all spawn at once;
* cap `OMP_NUM_THREADS` / `MKL_NUM_THREADS` at 6 (8 x 16 default threads oversubscribes
  32 logical cores and inflates per-process thread arenas);
* watch each trainer for CPU progress and kill + retry after `-HangSeconds 300`, cleaning
  the partial `monitor/` directory so the retry starts from a clean curve;
* then run `ladder_analysis.py` in the same job.

Healthy signature (4 in flight, from `Get-Process` deltas): each trainer accumulating
~5.7 CPU-seconds per wall second, free commit staying above ~14 GB.
