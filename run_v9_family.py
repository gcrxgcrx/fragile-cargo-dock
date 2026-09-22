"""Train the v9 family and score it on one fresh block.

Step 1 of the plan agreed at 21:20: measure the *real* success distribution of the v9 family,
because only 2 of its 8 candidates have been trained so far (cand_02 -> 58/60, cand_06 -> 0/60).

Procedure: train the six untrained candidates at 1.2 M steps, seed 0, three at a time (the
binding resource on this machine is the Windows commit limit, not RAM -
`runs/env_007/ladder_train/README.md`), then evaluate all eight on the fresh block
38000-38059 so that every number sits inside one block.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = r"D:\Code\python\research\llm_env_310\Scripts\python.exe"
CONFIG = "configs/env007_terminal_rule_pilot.yaml"
OUT = REPO / "runs/env_007/prompt_ladder_v9"
CANDIDATES = ["00", "01", "02", "03", "04", "05", "06", "07"]
ALREADY_TRAINED = {"02", "06"}
PARALLEL = 3


def summary_path(c: str) -> Path:
    return OUT / f"cand_{c}" / "training" / "training_summary.json"


def launch(c: str) -> subprocess.Popen:
    save = OUT / f"cand_{c}" / "training"
    save.mkdir(parents=True, exist_ok=True)
    log = open(REPO / f"runs/env_007/train_v9_family_{c}.log", "w")
    err = open(REPO / f"runs/env_007/train_v9_family_{c}.err.log", "w")
    cmd = [
        PY, "-m", "training.train_sb3_wrapper", "--config", CONFIG,
        "--reward", f"runs/env_007/prompt_ladder_v9/cand_{c}/reward_v1.py",
        "--run-name", f"v9_family_{c}", "--save-dir", str(save.relative_to(REPO)).replace("\\", "/"),
        "--total-timesteps", "1200000", "--eval-episodes", "20", "--seed", "0",
    ]
    env = {"OMP_NUM_THREADS": "6", "MKL_NUM_THREADS": "6"}
    import os
    e = dict(os.environ)
    e.update(env)
    return subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=err, env=e)


def main() -> None:
    todo = [c for c in CANDIDATES if c not in ALREADY_TRAINED and not summary_path(c).exists()]
    print(f"to train: {todo}", flush=True)
    running: list[tuple[str, subprocess.Popen]] = []
    while todo or running:
        while todo and len(running) < PARALLEL:
            c = todo.pop(0)
            p = launch(c)
            running.append((c, p))
            print(f"  launched cand_{c} pid={p.pid}", flush=True)
        still = []
        for c, p in running:
            rc = p.poll()
            if rc is None:
                still.append((c, p))
            else:
                ok = summary_path(c).exists()
                print(f"  cand_{c} finished rc={rc} summary={ok}", flush=True)
        running = still
        if running:
            import time
            time.sleep(20)
    print("all trainings done", flush=True)

    # evaluate everything on one block
    dirs = [f"runs/env_007/prompt_ladder_v9/cand_{c}/training" for c in CANDIDATES
            if summary_path(c).exists()]
    print(f"evaluating {len(dirs)} runs on block 38000-38059", flush=True)
    subprocess.run(
        [PY, "eval_pool.py", "--episodes", "60", "--seed-offset", "38000",
         "--out", "runs/env_007/prompt_ladder_v9/eval_family_block38000.json", *dirs],
        cwd=REPO, check=False)
    rows = json.loads((OUT / "eval_family_block38000.json").read_text(encoding="utf-8-sig"))
    rows.sort(key=lambda r: -r["success"])
    print(f"\n{'candidate':<40} {'success':>9} {'dock':>6} {'return':>9}", flush=True)
    for r in rows:
        print(f"{r['run_dir']:<40} {r['success']:>4}/{r['episodes']:<4} "
              f"{r['dock_rate']:>6.2f} {r['mean_return']:>9.1f}", flush=True)
    s = [r["success"] for r in rows]
    if len(s) > 1:
        m = sum(s) / len(s)
        sd = (sum((x - m) ** 2 for x in s) / (len(s) - 1)) ** 0.5
        print(f"\nfamily: n={len(s)} mean={m:.1f}/60 sd={sd:.1f} min={min(s)} max={max(s)}", flush=True)


if __name__ == "__main__":
    main()
