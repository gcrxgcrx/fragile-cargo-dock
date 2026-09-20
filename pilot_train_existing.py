"""Train a set of *already generated* reward files, in parallel.

Written for controlled A/B probes: the same reward file is trained twice, once
with the harness default per-step clip and once with a different clip, so the
only difference between the two arms is the clip.

Usage
-----
    python pilot_train_existing.py \
        --config configs/env007_terminal_rule_pilot_clip600.yaml \
        --src  runs/env_007/terminal_rule_pilot/seed_0/gen_00 \
        --dst  runs/env_007/terminal_rule_pilot_clip600/seed_0/gen_00 \
        --total-timesteps 1200000 --eval-episodes 20 --parallel 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def train_one(config_path: str, reward_path: Path, out_dir: Path, run_name: str,
              total_timesteps: int, eval_episodes: int, seed: int) -> tuple[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "training.train_sb3_wrapper",
        "--config", config_path,
        "--reward", str(reward_path),
        "--run-name", run_name,
        "--save-dir", str(out_dir),
        "--total-timesteps", str(total_timesteps),
        "--eval-episodes", str(eval_episodes),
        "--seed", str(seed),
    ]
    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    return run_name, proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--src", required=True, help="dir containing cand_*/reward_v1.py")
    ap.add_argument("--dst", required=True, help="dir to write cand_*/training into")
    ap.add_argument("--total-timesteps", type=int, default=1_200_000)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    cands = sorted(p for p in src.glob("cand_*/reward_v1.py"))
    if not cands:
        raise SystemExit(f"no cand_*/reward_v1.py under {src}")

    print(f"training {len(cands)} candidates, {args.parallel} at a time")
    print(f"  config : {args.config}")
    print(f"  step   : {args.total_timesteps:,}")
    print(f"  out    : {dst}")

    failures = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {}
        for reward_path in cands:
            name = reward_path.parent.name
            run_name = f"{dst.name}/{name}/training"
            out_dir = dst / name / "training"
            fut = pool.submit(train_one, args.config, reward_path, out_dir,
                              run_name, args.total_timesteps, args.eval_episodes, args.seed)
            futures[fut] = name
        for fut in as_completed(futures):
            name, rc = fut.result()
            status = "ok" if rc == 0 else f"FAILED rc={rc}"
            print(f"[{name}] {status}", flush=True)
            if rc != 0:
                failures.append(name)

    if failures:
        raise SystemExit(f"failed: {failures}")


if __name__ == "__main__":
    main()
