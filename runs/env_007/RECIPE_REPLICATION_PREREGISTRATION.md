# Pre-registration: does the `r09` recipe generalise to a second LLM candidate?

Written 20:40, before the three trainings were launched.

The recipe, measured on `L2/cand_00` (arm `r09_progress_x50_stream`: **23/60 = 38.3 %**,
dock 0.98, one-sided Fisher **p = 8.7e-09** against 0/60 — the first LLM-derived reward in
this project to deliver the crate):

1. raise the candidate's **own approach term** ~50× so that acting beats doing nothing;
2. pay a **dense +20/step payoff on the instantaneous success predicate** so that *settling*
   is profitable, not only approaching.

Replication target `L2/cand_14` (same skeleton: own approach term, instantaneous
`dock_state`, one-off +300/+30 events; but it returns early when `dock_state > 0.5`, paying
nothing during the hold — so the stream is inserted in **both** branches).

## Probe predictions, computed before training (no training involved)

`advantage_scale_probe.py` on the four reward files:

| reward | A (per step) |
|---|---:|
| `g02_recipe` (×50 + stream) | **+0.355** |
| `g01_stream` (stream only) | +0.027 |
| `cand_14` original / `g00_copy` | −0.0004 |

## Predictions, fixed now

| # | prediction |
|---|---|
| **R1** | `g00_copy` = **0/60, dock 0.00**, matching the original's measured 0/60 |
| **R2** | `g02_recipe` **> 0/60**; if the recipe generalises, at least **8/60 (13 %)** — above the previous best LLM result in the whole project (5/60 = 8.3 %) |
| **R3** | `g02 > g01`: the scale half is necessary (probe: A = 0.355 vs 0.027) |
| **R4** | `g01_stream` alone is the genuinely open case: A = +0.027 is small but of the same order as `probeE` (+0.017, which works). If `g01 > 0/60`, the scale half is **not** necessary for this candidate and the recipe reduces to "pay for settling" |
| **R5** | mechanism: in whichever arm succeeds, `success_event` `active_rate` becomes **> 0** |

Protocol unchanged: `configs/env007_terminal_rule_pilot.yaml`, `n_envs=6`, clip 20, 1.2M
steps, seed 0, fresh seeds **32000–32059**, 60 episodes.

Standing caveat: oracle-authored edits. This bounds what a repair operator could reach and
is not evidence that a search would find the edit.
