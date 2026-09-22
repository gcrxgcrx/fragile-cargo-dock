"""Launcher for the CREATE run with the fixed spec + the v9 reward-structure block.

Config : configs/env007_fragilecargo_create_v9.yaml
Protocol: 10 rounds x 3 M steps x 20 evaluation episodes, seed 0, one lineage.
          Same budget as the EUREKA baseline and as the previous CREATE run
          (`fragilecargo_create_v2`, 10 rounds x 3 M), so the two method runs are
          comparable at equal environment steps.

Why a launcher instead of `run_create_fragilecargo.sh`:
  * the API key is read from DSapi.txt at runtime and only ever lives in the child's
    environment — it is never printed, never logged and never written into the repo
    (the bash script requires it pre-exported, which is easy to leak into a transcript);
  * a transient API or network error must not throw away hours of completed rounds, so
    the launcher retries and resumes from the first round with no `training_summary.json`
    instead of restarting the lineage.

Everything the child prints goes to `runs/env_007/fragilecargo_create_v9.log`
(stderr to `.err.log`), both append-only across attempts.

Usage:
    python run_create_v9_full.py
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
KEY_FILE = Path(r"D:\Code\python\research\DSapi.txt")

CONFIG = "configs/env007_fragilecargo_create_v9.yaml"
PREFIX = "fragilecargo_create_v9"
ROUNDS = 10
SEED = 0

# The pipeline stopped its own run twice with adaptive stop rules:
#   * round 2 -> `stop_after_solved_drop_keep_best` (round 1 scored 279.63 >= target 250, so
#     `solved_seen` became true, and round 2 scoring 204.46 < target then triggered the stop);
#   * round 4 -> `stop_solved_no_improvement_keep_best`.
# Both stops are correct behaviour for CREATE as configured, but they leave most of the
# budgeted rounds untrained, and on this environment the quantity worth comparing against
# EUREKA is the distribution over draws, not the best of two (handoff §7.1: the v9 single-shot
# candidate `cand_02` scored 58/35/0 across three training seeds).
#
#   --no-early-stop      disable only the drop rule
#   --no-early-stop-all  disable every adaptive stop rule, i.e. train all 10 rounds
#
# Either flag is a deliberate change to the search protocol and the phases are reported
# separately in the findings, so no one has to guess which rounds ran under which rule.
NO_EARLY_STOP = "--no-early-stop" in sys.argv
NO_EARLY_STOP_ALL = "--no-early-stop-all" in sys.argv

RUN = REPO / "runs/env_007" / PREFIX
SEED_ROOT = RUN / f"seed_{SEED}"
LOG = REPO / "runs/env_007" / f"{PREFIX}.log"
ERR = REPO / "runs/env_007" / f"{PREFIX}.err.log"

MAX_ATTEMPTS = 40
RETRY_WAIT_SEC = 60


def completed_rounds() -> int:
    """Highest round whose training finished, 0 if none.

    A round counts as complete only when its training summary exists *and* parses *and* the
    policy file is there. The launcher is stopped by hand during development (a defect in
    the card, a fixed bug in a later stage), and without the model check a half-trained
    round would be accepted as done and silently reused.
    """
    done = 0
    for i in range(1, ROUNDS + 1):
        train_dir = SEED_ROOT / f"iter_{i:02d}" / "training"
        summary = train_dir / "training_summary.json"
        if not summary.exists() or not (train_dir / "model.zip").exists():
            break
        try:
            json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            break
        done = i
    return done


def build_cmd(resume_from: int) -> list[str]:
    cmd = [
        PY, "-m", "pipeline.run_iterative_experiment",
        "--config", CONFIG,
        "--prefix", PREFIX,
        "--seed", str(SEED),
        "--rounds", str(ROUNDS),
        "--total-timesteps", "3000000",
        "--eval-episodes", "20",
    ]
    if NO_EARLY_STOP_ALL:
        cmd += ["--no-early-stop-all"]
    elif NO_EARLY_STOP:
        cmd += ["--no-early-stop"]
    if resume_from > 1:
        cmd += ["--resume-from", str(resume_from)]
    return cmd


def main() -> int:
    if not KEY_FILE.exists():
        raise SystemExit(f"API key file not found: {KEY_FILE}")
    key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit(f"API key file is empty: {KEY_FILE}")

    env = dict(os.environ)
    env.update({
        "DEEPSEEK_API_KEY": key,
        "EUREKA_DEEPSEEK_API_KEY": key,
        "DEEPSEEK_THINKING": "disabled",
        "OMP_NUM_THREADS": "6",
        "MKL_NUM_THREADS": "6",
        "PYTHONUNBUFFERED": "1",
    })

    print(f"python   : {PY}")
    print(f"config   : {CONFIG}")
    print(f"prefix   : {PREFIX}  seed {SEED}  rounds {ROUNDS}  timesteps 3000000")
    print(f"run dir  : {SEED_ROOT}")
    print(f"log      : {LOG}")
    print(f"resume at: round {completed_rounds() + 1}", flush=True)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        resume_from = completed_rounds() + 1
        if resume_from > ROUNDS:
            print("all rounds already complete", flush=True)
            return 0

        cmd = build_cmd(resume_from)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n===== attempt {attempt} at {stamp} (resume-from {resume_from}) =====", flush=True)
        with open(LOG, "a", encoding="utf-8") as out, open(ERR, "a", encoding="utf-8") as err:
            out.write(f"\n===== attempt {attempt} at {stamp} (resume-from {resume_from}) =====\n")
            out.flush()
            rc = subprocess.call(cmd, cwd=REPO, stdout=out, stderr=err, env=env)
        print(f"attempt {attempt} exit {rc}; completed rounds now {completed_rounds()}", flush=True)

        if rc == 0:
            print("RUN COMPLETE", flush=True)
            return 0
        if attempt < MAX_ATTEMPTS:
            time.sleep(RETRY_WAIT_SEC)

    print("attempt budget exhausted", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
