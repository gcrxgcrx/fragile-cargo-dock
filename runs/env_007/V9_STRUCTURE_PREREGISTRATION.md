# PRE-REGISTRATION: v9 — give the generator the *structure*, keep every channel closed

Written before any v9 candidate was generated (20:29). Companion:
`prompts/eureka_01_initial_reward_v9.md`, generator `make_prompt_v9.py`,
diff `runs/env_007/v9_prompt.diff`.

## The measurement this is built on

The environment's own reward, fed **through the wrapper** at 1.2 M / clip 20 (i.e. exactly the
pipeline's per-step treatment), scored on the shared fresh block 38000-38059:

| seed | 0 | 1 | 2 | 3 | mean | sd |
|---|---:|---:|---:|---:|---:|---:|
| native reward (passthrough) | **52/60** | **47/60** | **56/60** | **54/60** | 52.3 | **4.0** |

For comparison, on the same block: the hand-written `control_v1` scores {0, 35, 43, 54}
(mean 33, sd 23); the best LLM candidate ever produced (v8 `cand_01`) scores **4/60**; every
other LLM candidate scores 0.

So the environment is *not* hostile to a well-structured reward — it is stable at ~85 % for the
one reward that has the right structure. And of the nine native terms, **only `roughness`
(and the `hard_hit` count that shares its signal) needs a quantity the contract forbids**
(`peak_impulse`); the other seven are exactly recoverable from `obs`/`action`. Code-level
re-expression of the native reward reaches 98.3 % (`probeC`); the same with an observation
proxy for the impulse term reaches 73.3 % (`probeD`).

**Hypothesis under test:** the LLM's failure is (at least partly) an *information* failure —
it has never been told what the reward's structure is. v9 tells it, and keeps every privileged
channel closed (`info` forbidden, no `peak_impulse`, function of `obs`/`action` only).

## The prompt change

v9 = v8 + one block that states the nine native terms with their semantics and weights, gives
per-term guidance on how each is recovered from observations (and, for `roughness`, tells the
model to build a contact × closing-speed proxy), and declares that block to take precedence
over the generic rules that used to forbid such terms. Nothing else changes; the observation
and action contract, the boundary guard, the module-state rules and the code constraints are
untouched. `v8` is left on disk unmodified.

## Protocol

`pilot_generate_only.py`, `prompts/eureka_01_initial_reward_v9.md`, same context
(`runs/env_007/terminal_rule_pilot/seed_0/context`), config
(`configs/env007_terminal_rule_pilot.yaml`), model (`deepseek-flash`) and `temperature=0.7` as
the v7/v8 arms. **N = 8**, output `runs/env_007/prompt_ladder_v9/`.

Then, in this order, so that no training is spent before the mechanical read:

1. **mechanical**: does the candidate implement the nine terms? (`analyze_family_shape.py` +
   `check_settled_stream.py` + a term-presence read of the emitted code)
2. **training**: the **two** candidates with the cleanest term structure, 1.2 M steps, seed 0,
   evaluated on **38000-38059** (the same block as the passthrough and v8 numbers above, so
   every comparison is inside one block).
3. `dock_entered` and success reported separately, plus `max consecutive stable steps`
   (`diagnose_settling.py`) and the realised per-episode component table.

## Predictions, fixed now

| # | prediction |
|---|---|
| **V9-1** | ≥ 6/8 candidates contain the full term set (`approach_cargo`, `progress`, `dock_enter`, `action_cost`, `time_cost`, `success`-streak, boundary) with per-metre / per-step magnitudes within 3x of the stated weights |
| **V9-2** | **at least one candidate reaches ≥ 20/60** on 38000-38059 (the v8 family's best was 4/60) |
| **V9-3** | that candidate's `dock_entered` is ≥ 0.6 and its `max consecutive stable steps` is well above the v8 candidate's 2.42 — i.e. the gain, if any, shows up in settling |
| **V9-4** | the family is **not** uniformly good: I expect **at most 4/8** to exceed 20/60, because the `roughness` proxy is the one term the model must invent and the measured cost of getting it wrong is the whole behaviour (`gx15`/`gx50`: dock → 0.00) |

## Verdict rules, fixed now

* **V9-1 holds and V9-2 holds** → the information hypothesis is **confirmed at the level of the
  prompt**: the ceiling was "the generator was never told the reward's structure".
  Then the next step is not a better search operator but the honest re-framing of the study:
  measure how much information (structure / coefficients / channels) is needed, and what the
  search adds on top of it.
* **V9-1 holds and V9-2 fails** (structure present, still ≤ 8/60) → information is **not**
  sufficient; combined with the earlier single-variable results, the remaining explanation is
  the operator's inability to turn a stated structure into working code, and the mechanism to
  test becomes the closed loop (`run_repair_loop` with rounds) rather than the prompt.
* **V9-1 fails** (the model ignores or mangles the given structure) → the generator cannot
  follow an explicit specification, which is the strongest form of the operator limitation and
  makes any prompt-level route dead.

## Standing caveats

* One training seed per candidate at 1.2 M. The seed spread measured on this block for
  hand-written arms is large (0-54/60), so a *single* candidate's number cannot be read as a
  family property; that is why V9-1 (mechanical, n = 8) and V9-4 (distribution) are part of the
  pre-registration rather than only the best number.
* v9 is authored by me from measurements; like v2-v8 it is a scaffold change and must be
  reported as part of the prompt lineage, not as a search result. What v9 changes is the
  *information given to the generator*, which is precisely the variable under study.
