"""Full EUREKA run with the fixed spec (the context fix from PIPELINE_CONTEXT_DIAGNOSIS.md).

Config: `configs/env007_fragilecargo_eureka_v9_fixedspec.yaml`
  * prompt  = prompts/eureka_01_initial_reward_v9.md      (the structure recipe)
  * spec    = envs/env_007/task_spec_anonymized_v2.yaml   (carries the dock geometry)
Protocol: 4 generations, population 4 (elite 2 x 1 + 1 child each), 3 M steps per candidate,
20 evaluation episodes, 4 candidates trained concurrently - identical to the run that used the
v1 spec, so the two are directly comparable.

The context (environment card + expert reward context) is already on disk at
runs/env_007/fragilecargo_eureka_v9fix/seed_0/context/, so the driver reuses it; the wrapper
retries on transient API/network errors, which killed the first attempt of the previous run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = r"D:\Code\python\research\llm_env_310\Scripts\python.exe"
KEY_FILE = Path(r"D:\Code\python\research\DSapi.txt")
PREFIX = "fragilecargo_eureka_v9fix"
CFG = "configs/env007_fragilecargo_eureka_v9_fixedspec.yaml"
RUN = REPO / "runs/env_007" / PREFIX
LOG = RUN.parent / f"{PREFIX}.log"
ERR = RUN.parent / f"{PREFIX}.err.log"

CMD = [PY, "-m", "pipeline.run_eureka_population", "--config", CFG, "--prefix", PREFIX,
       "--generations", "4", "--population-size", "4", "--elite-size", "2",
       "--children-per-parent", "1", "--total-timesteps", "3000000",
       "--eval-episodes", "20", "--parallel-candidates", "4", "--seed", "0"]


def main() -> None:
    ctx = RUN / "seed_0" / "context"
    have = [p.name for p in ctx.glob("*.md")] if ctx.exists() else []
    print(f"context present: {sorted(have)}", flush=True)
    if not (ctx / "environment_card.md").exists():
        raise SystemExit("no environment card")

    key = KEY_FILE.read_text(encoding="utf-8").strip()
    env = dict(os.environ)
    env.update({"EUREKA_DEEPSEEK_API_KEY": key, "DEEPSEEK_API_KEY": key,
                "DEEPSEEK_THINKING": "disabled", "OMP_NUM_THREADS": "6",
                "MKL_NUM_THREADS": "6"})

    for attempt in range(1, 13):
        stamp = time.strftime("%H:%M:%S")
        print(f"\n===== attempt {attempt} at {stamp} =====", flush=True)
        with open(LOG, "a", encoding="utf-8") as out, open(ERR, "a", encoding="utf-8") as err:
            out.write(f"\n===== attempt {attempt} at {stamp} =====\n")
            out.flush()
            rc = subprocess.call(CMD + (["--resume"] if attempt > 1 else []),
                                 cwd=REPO, stdout=out, stderr=err, env=env)
        print(f"attempt {attempt} exit {rc}", flush=True)
        if rc == 0 and (RUN / "seed_0" / "eureka_summary.md").exists():
            print("RUN COMPLETE", flush=True)
            return
        time.sleep(60)
    print("attempt budget exhausted", flush=True)


if __name__ == "__main__":
    main()
