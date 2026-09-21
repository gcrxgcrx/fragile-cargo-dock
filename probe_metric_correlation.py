"""Which training-free statistic actually tracks the known outcomes?

Motivation
----------
`compare_shaping_scale.py` produced a shaping:terminal ratio that looked like it ordered
the outcomes, but on n=1 trajectory and with a contradiction already visible (probeD at
ratio 9.2 scores 73.3 % while r09 at 16.2 scores 38.3 %). Before any of that is written
into a v8 prompt, it has to be tested against every arm whose outcome is known.

This script computes, for every reward with a measured fresh-60 outcome:

  A               per-step advantage of the scripted controllers over `idle`
                  (the same statistic as advantage_scale_probe.py)
  ratio           mean over trajectories of  (shaping return) / (terminal payoff),
                  shaping = sum of all steps except the final one of a SUCCESS trajectory,
                  terminal = the reward on that final step (clipped at 20)
  hover           mean per-step reward on the stable tail of a success trajectory,
                  SKIPPING the first stable step (i.e. what a policy earns by keeping a
                  settled state alive after its one-off event has already fired)
  dock_share      fraction of the dock-dwell steps that pay at least +1

and reports the Spearman correlation of each against fresh-60 success and against
`dock_entered`. It also repeats the known A / dock_entered contingency table.

Usage:
    python probe_metric_correlation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import trajectory_ranking_check as trc  # noqa: E402
from advantage_scale_probe import HEURISTIC, advantage  # noqa: E402

CLIP = 20.0
TAIL = 60           # stable steps used for the hover readout

# (label, reward path or None for the native passthrough, fresh-60 success, dock_entered)
POOL = [
    ("v7_cand_00", "runs/env_007/prompt_ladder_v7/cand_00/reward_v1.py", 0, 0.00),
    ("v7_cand_01", "runs/env_007/prompt_ladder_v7/cand_01/reward_v1.py", 3, 0.18),
    ("v7_cand_02", "runs/env_007/prompt_ladder_v7/cand_02/reward_v1.py", 0, 0.00),
    ("v7_cand_03", "runs/env_007/prompt_ladder_v7/cand_03/reward_v1.py", 0, 0.00),
    ("v7_cand_04", "runs/env_007/prompt_ladder_v7/cand_04/reward_v1.py", 0, 0.00),
    ("v7_cand_05", "runs/env_007/prompt_ladder_v7/cand_05/reward_v1.py", 0, 0.00),
    ("v7_cand_06", "runs/env_007/prompt_ladder_v7/cand_06/reward_v1.py", 0, 0.00),
    ("v7_cand_07", "runs/env_007/prompt_ladder_v7/cand_07/reward_v1.py", 0, 0.00),
    ("L2_cand_00", "runs/env_007/prompt_ladder/L2/cand_00/reward_v1.py", 0, 0.00),
    ("L2_cand_04", "runs/env_007/prompt_ladder/L2/cand_04/reward_v1.py", 0, 0.00),
    ("L2_cand_05", "runs/env_007/prompt_ladder/L2/cand_05/reward_v1.py", 0, 0.00),
    ("L2_cand_14", "runs/env_007/prompt_ladder/L2/cand_14/reward_v1.py", 0, 0.00),
    ("L0_cand_02", "runs/env_007/prompt_ladder/L0/cand_02/reward_v1.py", 0, 0.00),
    ("r09_x50_stream", "runs/env_007/repair_test/r09_progress_x50_stream/reward_v1.py", 23, 0.98),
    ("r04_x50_only", "runs/env_007/repair_test/r04_own_progress_x50/reward_v1.py", 0, 0.97),
    ("q01_x75", "runs/env_007/ridge_width/q01_progress_x75/reward_v1.py", 0, 0.07),
    ("q02_x100", "runs/env_007/ridge_width/q02_progress_x100/reward_v1.py", 31, 0.95),
    ("p02_x200", "runs/env_007/approach_ablation/p02_progress_x200/reward_v1.py", 0, 0.72),
    ("probeA", "runs/env_007/ablation_probe/probeA_plus_roughness.py", 24, None),
    ("probeB", "runs/env_007/ablation_probe/probeB_event_terminal.py", 27, None),
    ("probeD", "runs/env_007/ablation_probe/probeD_obs_gentleness.py", 44, None),
    ("probeE", "runs/env_007/ablation_probe/probeE_obs_only_target.py", 39, None),
    ("control_v1", "runs/env_007/control_obs_only/reward.py", 0, None),
]


def clip(v):
    return max(-CLIP, min(CLIP, v))


def load_trajectories(seeds=6):
    return trc.build_library(seeds)


def replay(fn, traj):
    out = []
    for i in range(len(traj["obs"])):
        r = fn(traj["obs"][i], traj["act"][i], traj["nxt"][i], float(traj["native"][i]), {}, 0.0)
        out.append(float(r[0]) if isinstance(r, (tuple, list)) else float(r))
    return np.asarray(out)


def metrics_for(path, library, success_traj, traj_by_ctrl):
    fn = trc.load_reward(path)
    out = {}

    # --- ratio: shaping vs terminal on successful trajectories ---------------
    ratios = []
    for traj in library:
        if not traj["success"] or len(traj["obs"]) < 5:
            continue
        r = replay(fn, traj)
        shaping = float(np.sum(np.clip(r[:-1], -CLIP, CLIP)))
        terminal = abs(float(np.clip(r[-1], -CLIP, CLIP)))
        ratios.append(shaping / max(1e-9, terminal))
    out["ratio"] = float(np.mean(ratios)) if ratios else float("nan")

    # --- hover: per-step pay for keeping a settled state alive ---------------
    # The stable tail of the success trajectory: replay those steps and skip the
    # first one (the step on which a one-off event fires).
    n = len(success_traj["obs"])
    start = max(1, n - TAIL - 1)
    vals = []
    for i in range(start, n - 1):
        r = fn(success_traj["obs"][i], success_traj["act"][i], success_traj["nxt"][i],
               float(success_traj["native"][i]), {}, 0.0)
        vals.append(clip(float(r[0]) if isinstance(r, (tuple, list)) else float(r)))
    out["hover"] = float(np.mean(vals[1:])) if len(vals) > 1 else float("nan")
    out["hover_n"] = len(vals)

    # --- dock_dwell: mean per-step reward over every inside-the-dock step ----
    dwell = []
    for traj in library:
        r = replay(fn, traj)
        inside = (np.abs(traj["nxt"][:, 12]) <= 0.024) & (np.abs(traj["nxt"][:, 13]) <= 0.030)
        if inside.any():
            dwell.extend(np.clip(r[inside], -CLIP, CLIP).tolist())
    out["dock_dwell"] = float(np.mean(dwell)) if dwell else float("nan")
    out["dock_share"] = float(np.mean(np.asarray(dwell) > 1.0)) if dwell else float("nan")
    out["dock_steps"] = len(dwell)

    # --- A: per-step advantage over idle (uses the same library) -------------
    per_step = {}
    for traj in library:
        r = replay(fn, traj)
        per_step.setdefault(traj["controller"], []).append(
            float(np.sum(np.clip(r, -CLIP, CLIP))) / max(1, len(traj["obs"])))
    act = float(np.mean([v for c in HEURISTIC for v in per_step[c]]))
    idle = float(np.mean(per_step["idle"]))
    out["A"] = act - idle
    return out


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return float("nan")
    x, y = x[ok], y[ok]
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom else float("nan")


def main() -> None:
    print("building the scripted-trajectory library (6 seeds x 9 controllers)...", flush=True)
    library = trc.build_library(6)
    succ = [t for t in library if t["success"] and len(t["obs"]) > TAIL + 5]
    if not succ:
        raise SystemExit("no successful trajectory in the library")
    success_traj = succ[0]
    print(f"library: {len(library)} trajectories, {len(succ)} successful; "
          f"using {success_traj['controller']} ({len(success_traj['obs'])} steps) "
          f"for the hover readout\n")

    rows = []
    for name, path, success, dock in POOL:
        if path is None:
            continue
        try:
            m = metrics_for(path, library, success_traj, None)
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<16} ERROR {type(exc).__name__}: {exc}")
            continue
        rows.append(dict(name=name, success=success, dock=dock, **m))

    hdr = f"{'arm':<16} {'succ/60':>8} {'dock':>6} {'A':>8} {'ratio':>8} {'hover':>8} {'dock_dwell':>11} {'dock_share':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: -(r["success"] or 0)):
        d = f"{r['dock']:.2f}" if r["dock"] is not None else "  -  "
        print(f"{r['name']:<16} {r['success']:>8} {d:>6} {r['A']:>8.3f} {r['ratio']:>8.2f} "
              f"{r['hover']:>8.3f} {r['dock_dwell']:>11.3f} {r['dock_share']:>11.3f}")

    print()
    stats = ["A", "ratio", "hover", "dock_dwell", "dock_share"]
    print(f"{'statistic':<14} {'rho vs success':>15} {'rho vs dock':>13}")
    print("-" * 46)
    summary = {}
    for s in stats:
        vals = [r[s] for r in rows]
        rs = spearman(vals, [r["success"] for r in rows])
        with_dock = [r for r in rows if r["dock"] is not None]
        rd = spearman([r[s] for r in with_dock], [r["dock"] for r in with_dock])
        summary[s] = {"rho_success": rs, "rho_dock": rd}
        print(f"{s:<14} {rs:>15.3f} {rd:>13.3f}")

    print()
    print("=== the A > 0 / dock_entered contingency table (where dock is known) ===")
    known = [r for r in rows if r["dock"] is not None]
    for label, sel in (("A <= 0", lambda r: r["A"] <= 0), ("A > 0", lambda r: r["A"] > 0)):
        sub = [r for r in known if sel(r)]
        docked = [r for r in sub if r["dock"] > 0]
        print(f"  {label:<8} n={len(sub):<2} dock_entered>0: {len(docked)}/{len(sub)}"
              f"   successes in this cell: {sum(r['success'] for r in sub)}"
              f"   max success: {max((r['success'] for r in sub), default=0)}")

    out = {"rows": rows, "spearman": summary, "clip": CLIP,
           "hover_definition": "mean per-step reward over the stable tail of a success "
                               "trajectory, excluding the first stable step"}
    p = Path("runs/env_007/metric_correlation.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
