"""Compare generation *paths* on the same prompt: does the pipeline's context break v9?

Reads three sets of `training_summary.json` and reports, per candidate:
  * the pipeline's own selection metric (`external_eval.mean_eval_reward`, 20 held-in episodes) - what the search optimises;
  * success terminations in that evaluation;
then evaluates the same candidates on one **fresh** block (`eval_pool.py`) so the comparison is
not made on the metric the pipeline itself used.

Sets compared:
  pipeline_context   runs/env_007/fragilecargo_eureka_v9/seed_0      (self-generated env card)
  pilot_context      runs/env_007/eureka_v9_pilotctx/seed_0          (pilot's card, same driver)
  single_shot        runs/env_007/prompt_ladder_v9                   (pilot_generate_only.py)

Usage:
    python compare_generation_paths.py [--seed-offset 39000] [--episodes 60]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = r"D:\Code\python\research\llm_env_310\Scripts\python.exe"
SETS = {
    "pipeline_context": REPO / "runs/env_007/fragilecargo_eureka_v9/seed_0",
    "pilot_context": REPO / "runs/env_007/eureka_v9_pilotctx/seed_0",
    "single_shot": REPO / "runs/env_007/prompt_ladder_v9",
}


def collect(root: Path):
    out = []
    for p in sorted(root.rglob("training_summary.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:  # noqa: BLE001
            continue
        e = s.get("external_eval") or {}
        out.append({
            "dir": str(p.parent.relative_to(REPO)).replace("\\", "/"),
            "score": e.get("mean_eval_reward"),
            "terminated": e.get("termination_breakdown", {}).get("terminated"),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-offset", type=int, default=39000)
    ap.add_argument("--episodes", type=int, default=60)
    args = ap.parse_args()

    everything = {}
    for name, root in SETS.items():
        rows = collect(root)
        everything[name] = rows
        print(f"=== {name} ({root.relative_to(REPO)}) : {len(rows)} trained candidates ===")
        for r in sorted(rows, key=lambda r: -(r["score"] or -999)):
            print(f"  {r['dir']:<64} score={r['score'] if r['score'] is None else round(r['score'],3):>9} "
                  f"terminated={r['terminated']}/20")
        if rows:
            sc = [r["score"] for r in rows if r["score"] is not None]
            print(f"  -> best score {max(sc):.2f}, mean {sum(sc)/len(sc):.2f}, "
                  f">=250 (delivery-level): {sum(1 for x in sc if x >= 250)}/{len(sc)}")
        print()

    dirs = [r["dir"] for rows in everything.values() for r in rows]
    if not dirs:
        return
    out_json = REPO / "runs/env_007/generation_path_compare.json"
    print(f"scoring {len(dirs)} candidates on fresh block {args.seed_offset}.."
          f"{args.seed_offset + args.episodes - 1}", flush=True)
    subprocess.run([PY, "eval_pool.py", "--episodes", str(args.episodes),
                    "--seed-offset", str(args.seed_offset), "--out", str(out_json), *dirs],
                   cwd=REPO, check=False)
    fresh = {}
    for r in json.loads(out_json.read_text(encoding="utf-8-sig")):
        fresh[r["run_dir"].replace("\\", "/")] = r["success"]
    print(f"\n{'set':<20} {'candidate':<58} {'score':>9} {'fresh/60':>9}")
    for name, rows in everything.items():
        for r in sorted(rows, key=lambda r: -(r["score"] or -999)):
            print(f"{name:<20} {r['dir'][-58:]:<58} "
                  f"{(r['score'] if r['score'] is not None else float('nan')):>9.2f} "
                  f"{str(fresh.get(r['dir'], '?')):>9}")
        sc = [r["score"] for r in rows if r["score"] is not None]
        fr = [fresh[d] for d in (r["dir"] for r in rows) if d in fresh]
        if sc:
            print(f"{'':<20} -> best score {max(sc):.2f}, "
                  f"fresh: max {max(fr) if fr else '?'} mean "
                  f"{(sum(fr)/len(fr)) if fr else float('nan'):.1f} "
                  f">=20/60: {sum(1 for x in fr if x >= 20)}/{len(fr)}")
        print()


if __name__ == "__main__":
    main()
