"""Build the C0/C1 isolation pair: `control_v1` with and without ONE added term.

Question (see `runs/env_007/CLOSE_SPEED_ISOLATION_PREREGISTRATION.md`):
`control_v1` scores 0/60 and `probeD` scores 73.3 %, and the two files differ by exactly
one term — an observation-only closing-speed penalty. This script materialises the pair
from `control_obs_only/reward.py` so the ONLY difference between the two arms is that
term, with auditable anchors (it refuses to patch blind).

    C0 = control_v1, body byte-identical
    C1 = C0 + the closing-speed term  (== probeD)

Usage:
    python make_closing_speed_variants.py
"""

from __future__ import annotations

import difflib
from pathlib import Path

SRC = Path("runs/env_007/control_obs_only/reward.py")
OUT = Path("runs/env_007/close_speed_test")

# the anchor: the components dict of control_v1, and the total line
ANCHOR = """    components = {
        "crate_progress": float(crate_progress),
        "cart_approach": float(cart_approach),
        "dock_settled_hold": float(settled_hold),
        "boundary": float(boundary),
        "shove": float(shove),
    }
    total = crate_progress + cart_approach + settled_hold + boundary + shove
"""

C1_BODY = '''    # ---- the ONE added term: an observation-only closing-speed penalty ------
    # Copied verbatim from ablation_probe/probeD_obs_gentleness.py. This is the
    # only difference between C0 and C1.
    cos_h = obs[2]
    sin_h = obs[3]
    cart_forward_speed = obs[4] * 3.0
    crate_speed_along_heading = cvx * cos_h + cvy * sin_h
    closing_speed = cart_forward_speed - crate_speed_along_heading
    if closing_speed < 0.0:
        closing_speed = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    gentleness = -0.05 * contact * closing_speed

    components = {
        "crate_progress": float(crate_progress),
        "cart_approach": float(cart_approach),
        "dock_settled_hold": float(settled_hold),
        "boundary": float(boundary),
        "shove": float(shove),
        "gentleness_obs": float(gentleness),
    }
    total = (crate_progress + cart_approach + settled_hold + boundary + shove + gentleness)
'''

HEADER = """# {name} — built by make_closing_speed_variants.py
#
# Base        : {src}
# Edit        : {edit}
# Everything else (both telescoping shaping terms, the +20 settled hold, the boundary
# guard, the shove penalty, every coefficient) is control_v1's own code, unmodified.
# Protocol and predictions: runs/env_007/CLOSE_SPEED_ISOLATION_PREREGISTRATION.md

"""


def main() -> None:
    src_text = SRC.read_text(encoding="utf-8")
    if ANCHOR not in src_text:
        raise SystemExit("anchor not found in control_v1; refusing to patch blind")

    # C0: byte-identical body, only a header is prepended
    c0 = HEADER.format(name="C0_control", src=SRC, edit="none (byte-identical control_v1 body)") + src_text
    # C1: exactly one block added
    c1_body = src_text.replace(ANCHOR, C1_BODY, 1)
    c1 = HEADER.format(name="C1_closing_speed", src=SRC,
                       edit="added the observation-only closing-speed penalty "
                            "(-0.05 * contact * closing_speed); nothing else") + c1_body

    for name, text in (("C0_control", c0), ("C1_closing_speed", c1)):
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "reward_v1.py").write_text(text, encoding="utf-8")
        print(f"  wrote {d / 'reward_v1.py'}")

    diff = difflib.unified_diff(c0.splitlines(), c1.splitlines(),
                                fromfile="C0_control", tofile="C1_closing_speed",
                                lineterm="", n=3)
    diff_text = "\n".join(diff) + "\n"
    (OUT / "C0_vs_C1.diff").write_text(diff_text, encoding="utf-8")
    added = [ln for ln in diff_text.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    print(f"\nC0 -> C1 adds {len(added)} lines:")
    for ln in added:
        print("   ", ln)
    print(f"wrote {OUT / 'C0_vs_C1.diff'}")


if __name__ == "__main__":
    main()
