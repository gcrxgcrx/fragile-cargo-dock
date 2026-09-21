"""Mechanical lint for one v8 prompt rule — LINT ONLY, not a success selector.

HISTORY / SCOPE LIMIT. This file's first version also tried to test "signed, cycle-neutral
progress" by moving the crate 0.1 m toward and away from the dock on a synthetic straight
line. That construction is **invalid**: with the cart and crate moved together, any change in
the crate-to-dock distance also changes the cart-to-crate distance, so `cart_approach` (a
legitimate telescoping term present in every working arm) contaminates the reading — it
reports +0.10/+0.10 for `probeD` and for `control_v1`, which are correct rewards. Isolating
the dock-progress term would require an observation in which the cart-to-crate distance is
unchanged while the crate-to-dock distance changes, i.e. a *physically unreachable* state —
exactly the failure mode that made the project's structural probes useless. So the signed
check is withdrawn; the loop-farming defect is measured on a *real* trained policy instead,
by summing the progress component over episodes and comparing it with the geometric bound
(`probe_progress_integral.py`).

What remains here is the one check that is both meaningful and deterministic:

CHECK B — settled payoff magnitude.
    Max per-step reward over the stable tail of a scripted success trajectory. If it is at
    (or within 1 % of) the harness clip (20), a one-off event paid on the same step adds
    nothing; if it is ~0, the candidate has no per-step payoff at all. Measured values in the
    v7 family are all exactly 20.00, which is why their +300 events are worth zero.

Usage:
    python check_v8_rules.py REWARD.py [REWARD.py ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trajectory_ranking_check as trc  # noqa: E402

CLIP = 20.0
HALF_X, HALF_Y = 5.0, 4.0
CART_HOME = (0.0, 0.0)
CART_COMBO = (1.0, 1.0)


def obs_vector(off_x_m, off_y_m, cart_xy=None, cart_fwd=1.0, contact=0.0):
    """Build a 19-D observation with the cart and the crate *moved together*.

    IMPORTANT: the cart must be placed so that the cart->crate vector is constant (the
    crate sits 0.4 m behind the cart in the body frame here). Otherwise a change in the
    crate-to-dock offset also changes the cart-to-crate distance, and the cart-approach
    term contaminates the signed-progress measurement.

    The crate's position is recovered by the environment as
    `cart_position + R(heading) * (obs[6]*3, obs[7]*3)`, so the cart is placed at
    `crate_position - (obs[6]*3, obs[7]*3)` with heading = +x. Only fields a reward may
    legitimately read are set; the rest are plausible constants.
    """
    rel_body = (-0.4, 0.0)                      # metres, body frame
    crate_xy = (off_x_m, off_y_m)               # the dock is at the origin
    if cart_xy is None:
        cart_xy = (crate_xy[0] - rel_body[0], crate_xy[1] - rel_body[1])
    o = np.zeros(19, dtype=np.float64)
    o[0] = cart_xy[0] / HALF_X
    o[1] = cart_xy[1] / HALF_Y
    o[2] = 1.0                 # cart heading = +x
    o[3] = 0.0
    o[4] = cart_fwd / 3.0
    o[5] = 0.0
    o[6] = rel_body[0] / 3.0   # body-frame crate offset
    o[7] = rel_body[1] / 3.0
    o[8] = 0.0                 # crate velocity (m/s) / 3
    o[9] = 0.0
    o[10] = 1.0                # crate aligned
    o[11] = 0.0
    o[12] = off_x_m / HALF_X   # crate-to-dock offset, the same normalisation the env uses
    o[13] = off_y_m / HALF_Y
    o[14] = contact
    o[15] = o[16] = o[17] = 0.0
    o[18] = 0.5
    return o


def call(fn, obs, nxt, act, native=0.0):
    out = fn(obs, act, nxt, native, {}, 0.0)
    total = float(out[0]) if isinstance(out, (tuple, list)) else float(out)
    comps = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else {}
    return total, (comps if isinstance(comps, dict) else {})


def check_b(fn, library):
    succ = [t for t in library if t["success"] and len(t["obs"]) > 15]
    if not succ:
        raise SystemExit("no successful trajectory in the library")
    traj = succ[0]
    n = len(traj["obs"])
    fn(traj["obs"][0], traj["act"][0], traj["nxt"][0], float(traj["native"][0]), {}, 0.0)
    tail = []
    for i in range(1, n):
        r, comps = call(fn, traj["obs"][i], traj["nxt"][i], traj["act"][i],
                        float(traj["native"][i]))
        tail.append((r, comps))
    vals = [max(-CLIP, min(CLIP, r)) for r, _ in tail[-12:]]
    return float(np.max(vals)), float(np.min(vals)), tail[-1][1]


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit("usage: check_v8_rules.py REWARD.py ...")
    library = trc.build_library(3)
    print("LINT ONLY — this check cannot predict learning.\n")
    hdr = f"{'candidate':<46} {'settled max':>12} {'settled min':>12} {'verdict':>8}"
    print(hdr)
    print("-" * len(hdr))
    for p in paths:
        name = p if len(p) < 46 else "..." + p[-43:]
        try:
            fn = trc.load_reward(p)
            b_max, b_min, _last = check_b(fn, library)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<46} ERROR {type(exc).__name__}: {exc}")
            continue
        verdict = "PASS" if b_max > 1.0 else "FAIL"
        note = ""
        if b_max >= CLIP - 0.05:
            note = "  <- at the clip: a one-off event on this step adds nothing"
        print(f"{name:<46} {b_max:>12.2f} {b_min:>12.2f} {verdict:>8}{note}")
    print()
    print("PASS = the candidate pays per step on a settled state (max over the stable tail > 1).")


if __name__ == "__main__":
    main()
