"""Supervise the CREATE v9 run to completion, then evaluate on a fresh seed block.

Started once and left running. It (1) prints a progress line whenever a round is scored,
(2) waits until every budgeted round has a scored training summary, (3) evaluates each round's
policy on a held-out block with the same code path as every other fresh-seed number in this
project (`eval_pool.py` -> `run_fragilecargo_baseline.load_trained_policy` / `evaluate`), and
(4) writes a compact result table. Rounds already present in the eval JSON are not re-evaluated.

Held-out blocks already consumed: 30000, 32000-39000, 40000. This uses 41000.

Usage:
    python tools/supervise_create_v9_run.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = r"D:\Code\python\research\llm_env_310\Scripts\python.exe"
PREFIX = "fragilecargo_create_v9"
SEED_ROOT = REPO / "runs/env_007" / PREFIX / "seed_0"
SUMMARY = SEED_ROOT / "experiment_summary.md"
FRESH_BLOCK = 41000
ROUNDS = 10
POLL_SEC = 120


def round_scores() -> list[dict]:
    out = []
    for path in sorted(SEED_ROOT.glob("iter_*/training/training_summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ext = data["external_eval"]
            comp = ext.get("final_policy_component_evaluation") or {}
            out.append({
                "iter": int(path.parents[1].name.split("_")[-1]),
                "score": ext["mean_eval_reward"],
                "terminated": ext["termination_breakdown"].get("terminated", 0),
                "episodes": ext.get("eval_episodes"),
                "duration_sec": data.get("train_duration_sec"),
                "components": {k: v.get("magnitude_share", 0.0) for k, v in comp.items()},
                "active": {k: v.get("active_rate", 0.0) for k, v in comp.items()},
            })
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
    return out


def finished(rounds: int) -> bool:
    """True once `rounds` rounds have a scored training summary.

    NOT `SUMMARY.exists()`: the lineage stopped itself after round 2 with
    `stop_after_solved_drop_keep_best` and wrote a summary then, so the summary's existence
    says nothing about whether the budgeted rounds are done. It is also re-written by every
    (re)start, so its `rounds_completed` field is not reliable either — the per-round training
    summaries are the durable record.
    """
    return len(round_scores()) >= rounds


def main() -> int:
    print(f"supervising {SEED_ROOT} (fresh block {FRESH_BLOCK}, target {ROUNDS} rounds)", flush=True)
    seen = 0
    while not finished(ROUNDS):
        rounds = round_scores()
        for rec in rounds[seen:]:
            top = sorted(rec["components"].items(), key=lambda kv: -kv[1])[:3]
            top_s = ", ".join(f"{k}={v:.2f}(act {rec['active'].get(k, 0):.2f})" for k, v in top)
            print(
                f"[{time.strftime('%H:%M:%S')}] iter {rec['iter']:02d}: score={rec['score']:.3f} "
                f"terminated={rec['terminated']}/{rec['episodes']} dur={rec['duration_sec']}s | {top_s}",
                flush=True,
            )
        seen = len(rounds)
        time.sleep(POLL_SEC)

    rounds = round_scores()
    for rec in rounds[seen:]:
        print(f"[{time.strftime('%H:%M:%S')}] iter {rec['iter']:02d}: score={rec['score']:.3f} "
              f"terminated={rec['terminated']}/{rec['episodes']}", flush=True)
    print(f"\n=== run finished at {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    print(SUMMARY.read_text(encoding="utf-8"), flush=True)

    # Evaluate every trained round on a fresh block.
    eval_json = REPO / "runs/env_007" / PREFIX / f"eval_block{FRESH_BLOCK}.json"
    trained = sorted(p.parent for p in SEED_ROOT.glob("iter_*/training/model.zip"))
    already = []
    if eval_json.exists():
        try:
            already = [Path(r["run_dir"]).resolve() for r in json.loads(eval_json.read_text(encoding="utf-8"))]
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            already = []
    pending = [d for d in trained if d.resolve() not in already]
    print(f"trained policies: {len(trained)}, already evaluated: {len(already)}, pending: {len(pending)}", flush=True)
    if pending:
        # Evaluate only the rounds that have no record yet, into a scratch file, then merge.
        # Re-running the finished rounds would cost the same block again for no new information,
        # and eval_pool.py has no append mode.
        scratch = eval_json.with_suffix(".pending.json")
        cmd = [
            PY, "eval_pool.py", "--episodes", "60", "--seed-offset", str(FRESH_BLOCK),
            "--out", str(scratch), "--root", str(SEED_ROOT), "--pattern", "iter_*/training/model.zip",
        ]
        print("$ " + " ".join(cmd), flush=True)
        with open(REPO / "runs/env_007" / f"{PREFIX}.eval.log", "a", encoding="utf-8") as log:
            rc = subprocess.call(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        print(f"eval exit {rc}", flush=True)
        if scratch.exists():
            merged = json.loads(scratch.read_text(encoding="utf-8"))
            scratch.unlink()
            if already:
                merged += [r for r in json.loads(eval_json.read_text(encoding="utf-8"))
                           if Path(r["run_dir"]).resolve() not in
                           {Path(m["run_dir"]).resolve() for m in merged}]
            merged.sort(key=lambda r: Path(r["run_dir"]).parts[-3])
            eval_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if eval_json.exists():
        records = json.loads(eval_json.read_text(encoding="utf-8"))
        records.sort(key=lambda r: -r["success"])
        table = ["", f"# CREATE v9 — fresh block {FRESH_BLOCK}-{FRESH_BLOCK + 59}", "",
                 "| policy | success | dock | mean return | reasons |", "|---|---:|---:|---:|---|"]
        for rec in records:
            table.append(
                f"| {Path(rec['run_dir']).parts[-3]} | {rec['success']}/60 | {rec['dock_rate']:.2f} "
                f"| {rec['mean_return']:.1f} | {rec['reasons']} |"
            )
        text = "\n".join(table) + "\n"
        (REPO / "runs/env_007" / PREFIX / f"eval_block{FRESH_BLOCK}.md").write_text(text, encoding="utf-8")
        print(text, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
