"""Approach-reward ablation: does ADDING a reward for getting near the dock help?

Motivation: the user's second proposal — "给接近泊位加奖励". This is not a side idea: the
measured two-edit repair that took two candidates from 0/60 to 23/60 has, as its **first**
edit, exactly a much stronger approach reward (the candidate's own `crate_to_dock_progress`
coefficient ×50). So the approach direction is already the measured fix, and the open questions
are about its **form** and its **size**:

* **delta vs state.** The working recipe pays for *reducing* the distance
  (`progress = d_now − d_next`, gated by proximity and alignment). The measured failure mode of
  the `L0` family was a **state** reward — `crate_docking_quality`, active 100 % of steps,
  +311 per episode, 0/60: a policy that hovers collects it. So a "being close" reward is the
  form with a known pathology, while a "getting closer" reward is the form with a known win.
* **how much.** `r04` (the ×50 approach edit alone) reaches `dock_entered` 0.97 and still scores
  0/60: more approach pull delivers the crate but does not make it *stay*. Whether even more
  pull helps, or tips into overshoot/violent contact, is untested.
* **the smooth funnel.** The best version of "reward for approaching" may be a dense term that
  pays for being near **and slow** — a smooth gradient into the settled state, which is the
  release-lead-time skill expressed continuously rather than as a cliff.

Base reward for every arm: `runs/env_007/repair_test/r09_progress_x50_stream/reward_v1.py` (23/60,
`dock_entered` 0.98). Shared matched baseline: arm `o00_base` in
`runs/env_007/overshoot_ablation/` (a byte-identical copy of it), scored on the SAME fresh block
35000–35059, so no cross-block comparison is needed.

Arms:

    p01_proximity        + dense reward for BEING near the dock       20/(1+30d)
    p02_progress_x200    the delta term 600 -> 2400 (200x the original 12)
    p03_x200_gentle      p02 + closing-speed penalty
    p04_proximity_gentle p01 + closing-speed penalty
    p05_funnel           + dense reward for being near AND slow: 20 * near * max(0, 1 - v/0.3)

Usage:
    python make_approach_variants.py \
        --src runs/env_007/repair_test/r09_progress_x50_stream/reward_v1.py \
        --out-dir runs/env_007/approach_ablation
"""

from __future__ import annotations

import argparse
from pathlib import Path

TOTAL_ANCHOR = "    total = 0.0\n"
PROGRESS_OLD = ("    components[\"crate_to_dock_progress\"] = "
                "600.0 * progress * near_gate * gate\n")
PROGRESS_NEW = ("    components[\"crate_to_dock_progress\"] = "
                "2400.0 * progress * near_gate * gate\n")

PROXIMITY = '''
    # ---- APPROACH (state form): dense reward for BEING near the dock ----
    # NOTE on scale: obs[12], obs[13] are normalised by FIELD_HALF_W/H (5.0 m, 4.0 m), and the
    # dock tolerance is 0.024 in those units, while the spawn distance is ~0.82. A proximity
    # term must therefore decay on a scale of ~0.05 units, not 1/30 -- the first version of this
    # file used 1/(1+30d) and the advantage probe showed it paying the DO-NOTHING controller
    # +0.73 per step (i.e. a near-constant bonus across the whole arena, not a proximity reward).
    _pd = (float(next_obs[12]) ** 2 + float(next_obs[13]) ** 2) ** 0.5
    components["APPROACH_proximity"] = float(20.0 * 2.718281828459045 ** (-_pd / 0.05))
'''

FUNNEL = '''
    # ---- APPROACH (funnel form): dense reward for being near AND slow ----
    # A smooth gradient towards the settled state; this is the release-lead-time skill
    # expressed continuously (pay only when close and not moving fast). Same scale note as
    # above: the decay constant is 0.05 normalised units (~0.25 m).
    _fd = (float(next_obs[12]) ** 2 + float(next_obs[13]) ** 2) ** 0.5
    _fnear = 2.718281828459045 ** (-_fd / 0.05)
    _fslow = 1.0 - crate_speed / 0.3
    if _fslow < 0.0:
        _fslow = 0.0
    components["APPROACH_funnel"] = float(20.0 * _fnear * _fslow)
'''

GENTLENESS = '''
    # ---- GENTLENESS: closing speed while in contact (the release-lead-time signal) ----
    _gcos_h, _gsin_h = obs[2], obs[3]
    _gcart_fwd = obs[4] * 3.0
    _galong = crate_vx * _gcos_h + crate_vy * _gsin_h
    _gclosing = _gcart_fwd - _galong
    if _gclosing < 0.0:
        _gclosing = 0.0
    components["REPAIR_gentleness"] = float(-0.05 * contact * _gclosing)
'''

VARIANTS = [
    dict(name="p01_proximity", mode="proximity", x200=False, gentleness=False,
         note="dense state reward for being near (the L0-family form, on a working base)"),
    dict(name="p02_progress_x200", mode=None, x200=True, gentleness=False,
         note="the delta form, 4x stronger than the measured repair (200x the original)"),
    dict(name="p03_x200_gentle", mode=None, x200=True, gentleness=True,
         note="p02 + the closing-speed penalty"),
    dict(name="p04_proximity_gentle", mode="proximity", x200=False, gentleness=True,
         note="p01 + the closing-speed penalty"),
    dict(name="p05_funnel", mode="funnel", x200=False, gentleness=False,
         note="dense reward for being near AND slow: the strongest form of the idea"),
]


def build(src_text: str, v: dict, src: str) -> str:
    text = src_text
    edits = []

    if v["x200"]:
        if PROGRESS_OLD not in text:
            raise SystemExit("progress anchor not found; refusing to patch blind")
        text = text.replace(PROGRESS_OLD, PROGRESS_NEW, 1)
        edits.append("crate_to_dock_progress 600.0 -> 2400.0")

    if v["mode"] == "proximity":
        if TOTAL_ANCHOR not in text:
            raise SystemExit("anchor 'total = 0.0' not found; refusing to patch blind")
        text = text.replace(TOTAL_ANCHOR, PROXIMITY + "\n" + TOTAL_ANCHOR, 1)
        edits.append("added APPROACH_proximity = 20/(1+30d)")

    if v["mode"] == "funnel":
        if TOTAL_ANCHOR not in text:
            raise SystemExit("anchor 'total = 0.0' not found; refusing to patch blind")
        text = text.replace(TOTAL_ANCHOR, FUNNEL + "\n" + TOTAL_ANCHOR, 1)
        edits.append("added APPROACH_funnel = 20 * near * max(0, 1 - v/0.3)")

    if v["gentleness"]:
        if TOTAL_ANCHOR not in text:
            raise SystemExit("anchor 'total = 0.0' not found; refusing to patch blind")
        text = text.replace(TOTAL_ANCHOR, GENTLENESS + "\n" + TOTAL_ANCHOR, 1)
        edits.append("closing-speed gentleness (-0.05 * contact * closing)")

    header = ("# APPROACH-ABLATION VARIANT {name} — generated by make_approach_variants.py\n"
              "#\n"
              "# Base reward : {src}  (the measured two-edit repair, 23/60)\n"
              "# Edits       : {edits}\n"
              "# Oracle-authored ablation; see runs/env_007/APPROACH_ABLATION_PREREGISTRATION.md\n\n")
    return header.format(name=v["name"], src=src, edits="; ".join(edits)) + text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    src_text = Path(args.src).read_text(encoding="utf-8")
    out = Path(args.out_dir)
    for v in VARIANTS:
        d = out / v["name"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "reward_v1.py").write_text(build(src_text, v, args.src), encoding="utf-8")
        print(f"  {v['name']:<22} {v['note']}")
    print(f"\nwrote {len(VARIANTS)} variants under {out}")


if __name__ == "__main__":
    main()
