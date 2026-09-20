"""Tabulate where a generated reward's mass actually goes in the FINAL policy.

Why this exists
---------------
The ladder's L2 candidates pass every structural check (one-off terminal event,
gentleness gap > 0, terminal dominance) and still train to 0/60 on fresh seeds. The
natural next question is where the reward mass ended up instead — the answer is in
`training_summary.json` (`external_eval.final_policy_component_evaluation`, and the last
`monitor_snapshots` entry), which is exactly the per-component `active_rate` /
`magnitude_share` table that SESSION_STATE.md §3g documents as living in EUREKA's
reflection.

Usage
-----
    python component_share_report.py runs/env_007/ladder_train/*/training_summary.json
    python component_share_report.py --out REPORT.md <summaries...>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarise(path: Path) -> dict:
    s = json.loads(path.read_text(encoding="utf-8"))
    ext = s.get("external_eval", {}) or {}
    term = ext.get("termination_breakdown", {}) or {}
    final = ext.get("final_policy_component_evaluation", {}) or {}

    active = {k: v for k, v in final.items()
              if isinstance(v, dict) and (v.get("active_rate") or 0) > 0}
    top = sorted(active.items(),
                 key=lambda kv: abs(kv[1].get("magnitude_share") or 0.0),
                 reverse=True)

    snaps = s.get("monitor_snapshots") or []
    last_snap = snaps[-1] if snaps else {}

    rated = ext.get("episode_rewards") or []
    return {
        "name": path.parent.name,
        "reward_path": s.get("reward_path"),
        "steps": s.get("total_timesteps"),
        "clip": s.get("reward_clip"),
        "mean_eval_reward": ext.get("mean_eval_reward"),
        "successes": sum(1 for r in rated if r > 0),   # not env success; kept only as a hint
        "termination": term,
        "n_active_components": len(active),
        "top_components": [(k, v.get("magnitude_share"), v.get("active_rate"),
                            v.get("episode_sum_mean")) for k, v in top[:4]],
        "success_event_active_rate": (final.get("success_event", {}) or {}).get("active_rate"),
        "mean_generated_reward_last_snapshot": last_snap.get("mean_generated_reward"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--out", default=None, help="also write the markdown table here")
    args = ap.parse_args()

    rows = [summarise(Path(p)) for p in args.summaries]

    lines = []
    lines.append("| candidate | mean eval reward | episode terminations | success_event active | "
                 "mean generated reward (last snapshot) | dominant components (share) |")
    lines.append("|---|---:|---|---:|---:|---|")
    for r in rows:
        term = r["termination"] or {}
        term_s = ", ".join(f"{k}={v}" for k, v in term.items()) or "-"
        comps = ", ".join(
            f"{k} {100.0 * (share or 0.0):.0f}%" for k, share, _a, _e in r["top_components"]
        ) or "(none active)"
        se = r["success_event_active_rate"]
        mre = r["mean_generated_reward_last_snapshot"]
        lines.append(
            f"| {r['name']} | {r['mean_eval_reward']:.2f} | {term_s} | "
            f"{'n/a' if se is None else f'{se:.4f}'} | "
            f"{'n/a' if mre is None else f'{mre:.3e}'} | {comps} |")

    table = "\n".join(lines)
    print(table)

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(table + "\n", encoding="utf-8")
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
