"""Run the EUREKA v9 population search, surviving transient API/network failures.

Why: the first attempt died at 22:55 with `openai.APIConnectionError: Connection error.` during
the generation of `gen_00/cand_03` - a transient network drop, but the driver has no retry, so
the whole run is lost. This wrapper relaunches the driver with `--resume` (which skips any
candidate whose reward file and score already exist) until it completes or the attempt budget is
exhausted, and it appends every attempt to one log.

The API key is read from the key file at runtime and never written anywhere.
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
LOG = REPO / "runs/env_007/eureka_v9_run.log"
ERR = REPO / "runs/env_007/eureka_v9_run.err.log"
MAX_ATTEMPTS = 12
RETRY_DELAY_S = 60

CMD = [PY, "-m", "pipeline.run_eureka_population",
       "--config", "configs/env007_fragilecargo_eureka_v9.yaml",
       "--prefix", "fragilecargo_eureka_v9",
       "--generations", "4", "--population-size", "4", "--elite-size", "2",
       "--children-per-parent", "1", "--total-timesteps", "3000000",
       "--eval-episodes", "20", "--parallel-candidates", "4", "--seed", "0"]


def main() -> None:
    key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key.startswith("sk-"):
        raise SystemExit("key file invalid")
    env = dict(os.environ)
    env.update({"EUREKA_DEEPSEEK_API_KEY": key, "DEEPSEEK_API_KEY": key,
                "DEEPSEEK_THINKING": "disabled", "OMP_NUM_THREADS": "6",
                "MKL_NUM_THREADS": "6"})

    for attempt in range(1, MAX_ATTEMPTS + 1):
        stamp = time.strftime("%H:%M:%S")
        print(f"\n===== attempt {attempt}/{MAX_ATTEMPTS} at {stamp} =====", flush=True)
        cmd = CMD + (["--resume"] if attempt > 1 else [])
        with open(LOG, "a", encoding="utf-8") as out, open(ERR, "a", encoding="utf-8") as err:
            out.write(f"\n===== attempt {attempt} at {stamp} =====\n")
            out.flush()
            rc = subprocess.call(cmd, cwd=REPO, stdout=out, stderr=err, env=env)
        print(f"attempt {attempt} exit code {rc}", flush=True)
        # the driver writes eureka_summary.md only on a clean finish
        summary = REPO / "runs/env_007/fragilecargo_eureka_v9/seed_0/eureka_summary.md"
        if rc == 0 and summary.exists():
            print("RUN COMPLETE", flush=True)
            return
        print(f"retrying in {RETRY_DELAY_S}s", flush=True)
        time.sleep(RETRY_DELAY_S)
    print("attempt budget exhausted", flush=True)


if __name__ == "__main__":
    main()
