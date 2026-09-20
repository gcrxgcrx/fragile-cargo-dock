"""Build the "repair" variants of an LLM candidate by a *localized structural addition*.

Question this serves
--------------------
The best structurally-compliant LLM candidates fail at 0/60 with their decisive terms
never active (`runs/env_007/LADDER_FINDINGS.md` §2). Two rival explanations:

  (a) **scale**: the terms are right but too small relative to everything else;
  (b) **reachability**: the terms are gated on states the policy never visits, so the
      realised reward is ~0 and there is no gradient to climb.

The measured evidence favours (b) — `L2/cand_00`'s own `crate_to_dock_progress` has
`active_rate = 0.0089` and contributes +0.0047 per episode — and under the harness's
per-step clip of 20, *raising* a one-off event cannot help either (a +300 event is clipped
to +20, exactly what a dense +20/step term earns per step). So the test is whether a
*dense, unconditionally active* guidance backbone (the thing probe D shows is sufficient:
73.3 % with control v1 + an obs-only gentleness term) can connect the policy to the
candidate's own terminal event.

What it produces
----------------
For each variant, a self-contained reward file whose body is the candidate's source with a
guidance block inserted verbatim before the total is summed, and the guidance keys added to
the existing "completion state zeroes the shaping terms" block. Same module-level state,
same thresholds, same event magnitudes. Nothing else is touched, and every variant file
carries a header recording exactly what was inserted.

The guidance formulas are copied from `ablation_probe/probeD_obs_gentleness.py` (observation
-only: `obs[12]/[13]` crate-to-dock offset, `obs[6]/[7]` cart-to-crate gap, `obs[2]/[3]`
heading, `obs[4]` cart forward speed, `obs[14]` contact flag). No `info` is read.

  crate_progress  = (|offset_t| - |offset_t+1|) * 5.0   # metres along x
  cart_approach   = (|gap_t|    - |gap_t+1|   )          # metres
  gentleness      = -0.05 * contact * max(0, cart_forward - crate_velocity·heading)

This is a HUMAN-ORACLE upper bound, in the sense of SESSION_STATE.md §4 error #7: it
measures whether a repairable target *exists*, not whether a search could find the edit.

Usage
-----
    python make_repair_variants.py \
        --src runs/env_007/prompt_ladder/L2/cand_00/reward_v1.py \
        --out-dir runs/env_007/repair_test
"""

from __future__ import annotations

import argparse
from pathlib import Path

GUIDANCE = '''
    # ---- REPAIR: dense, unconditionally active guidance (copied from probe D) ----
    # Inserted by make_repair_variants.py. Observation-only; reads no `info`.
    _gx0, _gy0 = obs[12] * 5.0, obs[13] * 4.0
    _gx1, _gy1 = next_obs[12] * 5.0, next_obs[13] * 4.0
    _gdist_prev = (_gx0 * _gx0 + _gy0 * _gy0) ** 0.5
    _gdist_now = (_gx1 * _gx1 + _gy1 * _gy1) ** 0.5
    components["REPAIR_crate_progress"] = float({w_progress} * (_gdist_prev - _gdist_now))

    _grx0, _gry0 = obs[6] * 3.0, obs[7] * 3.0
    _grx1, _gry1 = next_obs[6] * 3.0, next_obs[7] * 3.0
    _ggap_prev = (_grx0 * _grx0 + _gry0 * _gry0) ** 0.5
    _ggap_now = (_grx1 * _grx1 + _gry1 * _gry1) ** 0.5
    components["REPAIR_cart_approach"] = float({w_approach} * (_ggap_prev - _ggap_now))
'''

GENTLENESS = '''
    _gcos_h, _gsin_h = obs[2], obs[3]
    _gcart_fwd = obs[4] * 3.0
    _galong = crate_vx * _gcos_h + crate_vy * _gsin_h
    _gclosing = _gcart_fwd - _galong
    if _gclosing < 0.0:
        _gclosing = 0.0
    components["REPAIR_gentleness"] = float(-0.05 * contact * _gclosing)
'''

ZERO_ANCHOR = '        components["obstacle_penalty"] = 0.0\n'
ZERO_ADD = ('        components["REPAIR_crate_progress"] = 0.0\n'
            '        components["REPAIR_cart_approach"] = 0.0\n'
            '        if "REPAIR_gentleness" in components:\n'
            '            components["REPAIR_gentleness"] = 0.0\n')

TOTAL_ANCHOR = '    total = 0.0\n'

VARIANTS = [
    dict(name="r00_copy", w_progress=0.0, w_approach=0.0, gentleness=False, guidance=False,
         zero_events=False, progress_coef=None,
         note="exact copy of the base candidate: machinery control, must reproduce 0/60"),
    dict(name="r01_guidance", w_progress=1.0, w_approach=1.0, gentleness=False, guidance=True,
         zero_events=False, progress_coef=None,
         note="base + dense guidance at probe D's natural weights"),
    dict(name="r02_guidance_gentle", w_progress=1.0, w_approach=1.0, gentleness=True,
         guidance=True, zero_events=False, progress_coef=None,
         note="r01 + probe D's obs-only gentleness term"),
    dict(name="r03_guidance_x3", w_progress=3.0, w_approach=3.0, gentleness=False,
         guidance=True, zero_events=False, progress_coef=None,
         note="r01 with the guidance backbone at 3x weight"),
    dict(name="r04_own_progress_x50", w_progress=0.0, w_approach=0.0, gentleness=False,
         guidance=False, zero_events=False, progress_coef=600.0,
         note="SCALE HYPOTHESIS: the candidate's own progress coefficient 12 -> 600, nothing added"),
    dict(name="r05_guidance_no_events", w_progress=1.0, w_approach=1.0, gentleness=True,
         guidance=True, zero_events=True, progress_coef=None, settled_stream=False,
         note="r02 with the candidate's own terminal events zeroed: does its payoff still matter?"),
    dict(name="r06_settled_stream", w_progress=0.0, w_approach=0.0, gentleness=False,
         guidance=False, zero_events=False, progress_coef=None, settled_stream=True,
         note="base + a DENSE per-step payoff on the same success predicate (+20/step), i.e. "
              "the one-off event converted into a stream, as every working hand-written arm has"),
    dict(name="r07_stream_and_guidance", w_progress=1.0, w_approach=1.0, gentleness=True,
         guidance=True, zero_events=False, progress_coef=None, settled_stream=True,
         note="r02 + the dense payoff stream"),
    # --- wave 3: all built on r04's finding (progress x50 delivers the crate, 97 % dock
    #     entry, still 0/60 because the reward pays for APPROACHING and the env requires
    #     SETTLING). Each arm adds exactly one settling lever to r04. ---
    dict(name="r08_progress_x50_gentle", w_progress=0.0, w_approach=0.0, gentleness=True,
         guidance=False, zero_events=False, progress_coef=600.0, settled_stream=False,
         note="r04 + probe D's obs-only gentleness (a settling PENALTY)"),
    dict(name="r09_progress_x50_stream", w_progress=0.0, w_approach=0.0, gentleness=False,
         guidance=False, zero_events=False, progress_coef=600.0, settled_stream=True,
         note="r04 + a dense settled PAYOFF stream (+20/step on done_cond)"),
    dict(name="r10_x50_stream_gentle", w_progress=0.0, w_approach=0.0, gentleness=True,
         guidance=False, zero_events=False, progress_coef=600.0, settled_stream=True,
         note="r04 + gentleness + settled payoff: both settling levers"),
    dict(name="r11_x50_own_speedpen_x100", w_progress=0.0, w_approach=0.0, gentleness=False,
         guidance=False, zero_events=False, progress_coef=600.0, settled_stream=False,
         speed_pen_coef=-300.0,
         note="r04 + the candidate's OWN dock_speed_penalty coefficient -3.0 -> -300.0"),
]

SPEED_PEN_LINE = '    components["dock_speed_penalty"] = -3.0 * near_dock * over_speed\n'

SETTLED_STREAM_BLOCK = '''
    # ---- REPAIR: the one-off payoff, converted into a DENSE per-step stream ----
    # Paid on the candidate's own success predicate (done_cond: inside & aligned & slow),
    # the same +20/step that probe D's `dock_settled_hold` uses. Under the per-step clip of
    # 20 a one-off event can never be worth more than one step of such a stream.
    components["REPAIR_settled_stream"] = float(20.0 * done_cond)
'''

PROGRESS_LINE = ('    components["crate_to_dock_progress"] = 12.0 * progress * near_gate * gate\n')

ZERO_EVENTS_BLOCK = '''
    # ---- REPAIR ATTRIBUTION: the base candidate's own terminal events, removed ----
    components["enter_dock_event"] = 0.0
    components["success_event"] = 0.0
'''


def build(src_text: str, variant: dict, src: str) -> str:
    text = src_text
    edits = []

    if variant["progress_coef"] is not None:
        if PROGRESS_LINE not in text:
            raise SystemExit("progress-coefficient anchor not found; refusing to patch blind")
        text = text.replace(
            PROGRESS_LINE,
            '    components["crate_to_dock_progress"] = {c} * progress * near_gate * gate\n'.format(
                c=variant["progress_coef"]), 1)
        edits.append(f"crate_to_dock_progress coefficient 12.0 -> {variant['progress_coef']}")

    if variant.get("speed_pen_coef") is not None:
        if SPEED_PEN_LINE not in text:
            raise SystemExit("dock_speed_penalty anchor not found; refusing to patch blind")
        text = text.replace(
            SPEED_PEN_LINE,
            '    components["dock_speed_penalty"] = {c} * near_dock * over_speed\n'.format(
                c=variant["speed_pen_coef"]), 1)
        edits.append(f"dock_speed_penalty coefficient -3.0 -> {variant['speed_pen_coef']}")

    # gentleness is independent of the guidance backbone (wave 3 needs it standalone)
    gentle_block = GENTLENESS if variant["gentleness"] else ""
    if variant["guidance"]:
        block = GUIDANCE.format(w_progress=variant["w_progress"],
                                w_approach=variant["w_approach"]) + gentle_block
        if TOTAL_ANCHOR not in text:
            raise SystemExit("anchor 'total = 0.0' not found; refusing to patch blind")
        if ZERO_ANCHOR not in text:
            raise SystemExit("completion-zeroing anchor not found; refusing to patch blind")
        text = text.replace(TOTAL_ANCHOR, block + "\n" + TOTAL_ANCHOR, 1)
        text = text.replace(ZERO_ANCHOR, ZERO_ANCHOR + ZERO_ADD, 1)
        edits.append(
            f"added dense guidance (progress x{variant['w_progress']}, "
            f"approach x{variant['w_approach']}"
            + (", gentleness -0.05)" if variant["gentleness"] else ")"))
    elif variant["gentleness"]:
        if TOTAL_ANCHOR not in text:
            raise SystemExit("anchor 'total = 0.0' not found; refusing to patch blind")
        text = text.replace(TOTAL_ANCHOR, gentle_block + "\n" + TOTAL_ANCHOR, 1)
        edits.append("added obs-only gentleness -0.05 (standalone)")

    if variant.get("settled_stream"):
        if TOTAL_ANCHOR not in text:
            raise SystemExit("anchor 'total = 0.0' not found; refusing to patch blind")
        text = text.replace(TOTAL_ANCHOR, SETTLED_STREAM_BLOCK + "\n" + TOTAL_ANCHOR, 1)
        edits.append("added a dense +20/step payoff stream on the candidate's own done_cond")

    if variant["zero_events"]:
        if TOTAL_ANCHOR not in text:
            raise SystemExit("anchor 'total = 0.0' not found; refusing to patch blind")
        text = text.replace(TOTAL_ANCHOR, ZERO_EVENTS_BLOCK + "\n" + TOTAL_ANCHOR, 1)
        edits.append("zeroed the candidate's own enter_dock_event and success_event")

    if not edits:
        edits.append("no change (copy control)")

    header = ("# REPAIR VARIANT {name} — generated by make_repair_variants.py\n"
              "#\n"
              "# Base candidate : {src}\n"
              "# Edits          : {edits}\n"
              "# Everything else (module state, gating, thresholds, remaining magnitudes) is the\n"
              "# candidate's own code, unmodified. This is an oracle-authored edit: it bounds what\n"
              "# a repair operator could reach, and is NOT evidence that a search would find it.\n"
              "# See runs/env_007/REPAIR_TEST_PREREGISTRATION.md for the fixed predictions.\n\n")
    return header.format(name=variant["name"], src=src, edits="; ".join(edits)) + text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--only", default=None,
                    help="comma-separated variant names to write; default writes all. Use this "
                         "to add arms without rewriting files that are currently training.")
    args = ap.parse_args()

    src_path = Path(args.src)
    src_text = src_path.read_text(encoding="utf-8")
    out_root = Path(args.out_dir)

    wanted = set(args.only.split(",")) if args.only else None
    written = 0
    for v in VARIANTS:
        if wanted is not None and v["name"] not in wanted:
            continue
        d = out_root / v["name"]
        d.mkdir(parents=True, exist_ok=True)
        text = build(src_text, v, args.src)
        (d / "reward_v1.py").write_text(text, encoding="utf-8")
        print(f"  {v['name']:<26} {v['note']}")
        written += 1

    print(f"\nwrote {written} variants under {out_root}")


if __name__ == "__main__":
    main()
