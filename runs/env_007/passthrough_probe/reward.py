"""DIAGNOSTIC PROBE - not a search candidate, and deliberately rule-breaking.

This file returns the environment's own reward (`original_reward`), which the
search contract forbids. It exists only to isolate one variable:

    the native reward reaching 96.8 % is measured with `reward_fn = None`,
    i.e. it BYPASSES RewardOverrideWrapper entirely, so it is never clipped
    and never passed through the wrapper's error/scale path;

    every reward the search produces goes THROUGH the wrapper, which clips the
    per-step reward to `reward_clip` (default 20.0) - while the native reward's
    completion term is +300.

Feeding the native reward through the wrapper therefore isolates the clip:

    clip = 20   -> the +300 completion spike is squashed to +20
    clip = 600  -> non-binding, the native reward is reproduced exactly

The hand-written observation-only control in ../control_obs_only/ plateaus at
6.7 % on fresh seeds, statistically the same as the best LLM candidate (8.3 %),
against the native reward's 96.8 %. So the LLM's reward-design ability is not
the binding constraint. This probe asks whether the harness is.

Must never be added to any population, lineage or elite set.
"""


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    return float(original_reward), {"native_passthrough": float(original_reward)}
