"""Seed-replicate the two best v9 candidates.

Before any README sentence or pipeline claim is written, the numbers that matter are the
per-arm mean and spread of the two candidates that reached the native reward's level, not their
best draw. `cand_02` seed 0 scored 58/60 and seed 2 scored 0/60 (`training_s1` = 35/60), so the
spread is known to be large and the mean is unknown.

This runs seeds 1 and 2 for `cand_00` (seed 0 = 55/60) and seeds 1 and 2 for `cand_02`
(seed 0 = 58/60; `cand_02` seed 1 and 2 already exist from the earlier replication, so only
`cand_00` is genuinely new - the script skips anything already trained), then evaluates every
(candidate, seed) on the block 38000-38059.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = r"D:\Code\python\research\llm_env_310\Scripts\python.exe"
CONFIG = "configs/env007_terminal_rule_pilot.yaml"
OUT = REPO / "runs/env_007/prompt_ladder_v9"
JOBS = [("00", 1), ("00", 2)]          # cand_02 seeds 1,2 already exist
PARALLEL = 2


def save_dir(c: str, s: int) -> Path:
    return OUT / f"cand_{c}" / (f"training_s{s}" if s else "training")


def launch(c: str, s: int) -> subprocess.Popen:
    d = save_dir(c, s)
    d.mkdir(parents=True, exist_ok=True)
    log = open(REPO / f"runs/env_007/train_v9_c{c}_s{s}.log", "w")
    err = open(REPO / f"runs/env_007/train_v9_c{c}_s{s}.err.log", "w")
    cmd = [PY, "-m", "training.train_sb3_wrapper", "--config", CONFIG,
           "--reward", f"runs/env_007/prompt_ladder_v9/cand_{c}/reward_v1.py",
           "--run-name", f"v9_c{c}_s{s}", "--save-dir", str(d.relative_to(REPO)).replace("\\", "/"),
           "--total-timesteps", "1200000", "--eval-episodes", "20", "--seed", str(s)]
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "6", "MKL_NUM_THREADS": "6"})
    return subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=err, env=env)


def main() -> None:
    todo = [(c, s) for c, s in JOBS if not (save_dir(c, s) / "training_summary.json").exists()]
    running = []
    while todo or running:
        while todo and len(running) < PARALLEL:
            c, s = todo.pop(0)
            p = launch(c, s)
            running.append((c, s, p))
            print(f"  launched cand_{c} seed {s} pid={p.pid}", flush=True)
        still = []
        for c, s, p in running:
            if p.poll() is None:
                still.append((c, s, p))
            else:
                print(f"  cand_{c} seed {s} rc={p.returncode} "
                      f"summary={(save_dir(c,s)/'training_summary.json').exists()}", flush=True)
        running = still
        if running:
            time.sleep(20)

    dirs = []
    for c in ("00", "02"):
        for s in (0, 1, 2):
            d = save_dir(c, s)
            if (d / "training_summary.json").exists():
                dirs.append(str(d.relative_to(REPO)).replace("\\", "/"))
    print(f"evaluating {len(dirs)} runs on block 38000-38059", flush=True)
    subprocess.run([PY, "eval_pool.py", "--episodes", "60", "--seed-offset", "38000",
                    "--out", "runs/env_007/prompt_ladder_v9/eval_top2_seeds.json", *dirs],
                   cwd=REPO, check=False)
    rows = json.loads((OUT / "eval_top2_seeds.json").read_text(encoding="utf-8-sig"))
    for c in ("00", "02"):
        sub = [r for r in rows if f"cand_{c}/" in r["run_dir"].replace("\\", "/")]
        sub.sort(key=lambda r: r["run_dir"])
        vals = [r["success"] for r in sub]
        if not vals:
            continue
        m = sum(vals) / len(vals)
        sd = (sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
        print(f"cand_{c}: {vals}  mean={m:.1f}/60  sd={sd:.1f}", flush=True)
    print("\nreference: native reward through the wrapper = 52/47/56/54 on this block "
          "(mean 52.3, sd 3.9)", flush=True)


if __name__ == "__main__":
    main()
