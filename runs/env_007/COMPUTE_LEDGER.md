# COMPUTE LEDGER — measured throughput, and what each planned step costs

Recorded so a context reset does not lose the planning numbers. All throughput figures are
**measured on this machine during the 2026-09-20/21 session** (95+ trainings).

## Throughput (measured)

| unit | measured cost |
|---|---|
| 1.2M-step training (PPO, `n_envs=6`, clip 20) | ~11 min in a 4-parallel wave → **~2.75 min of wall clock per run** amortised |
| 0.6M-step training | ~6 min per 4-parallel wave |
| 3.0M-step training | ~27.5 min per run; ~28 min per 4-parallel wave |
| fresh-seed evaluation, 60 episodes | ~30 s per trained policy |
| one reward generation (`deepseek-flash`, T = 0.7) | ~6–10 s per candidate |

**Hard constraint:** free **commit** (page file), not RAM, is what binds. Keep concurrent
trainings ≤ 4; the queue's `-MinFreeCommitGB 3.0` guard handles it, and the failure mode when it
is ignored is worker death during `import torch` (`WinError 1455`) with the parent hanging at
0 CPU — see `runs/env_007/ladder_train/README.md`.

## Costs of the things on the table

| step | compute | wall clock |
|---|---:|---|
| ridge-width bisection (2 arms, running) | 2 × 1.2M | ~12 min |
| v8 prompt pilot (8–16 candidates, 1.2M) | 8–16 × 1.2M | 25–50 min |
| **operator ablation, one round** (2 operators × 2 seeds × 4 repairs) | 16 × 1.2M | **~50 min** + ~5 min generation |
| channel crossed in as well (real × sham × 2 operators) | 32 × 1.2M | ~1.7 h |
| full **EUREKA** pipeline at 3M (10 runs, 4-parallel) | 30M steps | **~1.5 h** |
| full **CREATE** pipeline at 3M (10 rounds, **single lineage → serial**) | 30M steps | **~5 h** |
| ⚠️ either full run with the config default `multi_seed.num_seeds: 3` | ×3 | **EUREKA ~4.5 h, CREATE ~14 h** |
| full pipelines at **1.2M** instead (comparable to the whole 95-model evidence base) | 2 × 10 runs | EUREKA ~45 min, CREATE ~1.8 h → **~2.5 h both** |

## Structural facts that drive the cost asymmetry

* **CREATE is single-lineage**, so its 10 rounds are 10 *sequential* 3M trainings — it cannot be
  amortised the way EUREKA's population can, even though both spend 30M environment steps. That
  is a property of the method, not of the machine.
* **Pin the seed count before any full run.** The base config sets `multi_seed.num_seeds: 3`; the
  paper's env_007 tables must be checked against that before spending 3× the compute.
* **3M vs 1.2M is a comparability decision, not just a cost one**: every measurement in
  `SESSION_STATE.md` §5 is at **1.2M**, so a 3M full run cannot be compared against that
  evidence base without re-baselining.
