"""Control experiment: is the pipeline's *self-generated context* what breaks v9 generation?

Measured facts this isolates
----------------------------
Same prompt (`prompts/eureka_01_initial_reward_v9.md`), two generation paths, same entry point
(`pipeline/run_eureka_population.py`, one API call per candidate):

* path PILOT: `pilot_generate_only.py`, which injects the static task spec + masked step source
  -> candidates scored **294-310** by the pipeline's own selection metric (19/20 success
  terminations at 1.2 M), and 2 of 8 family members reached 91.7-96.7 % on a fresh block;
* path PIPELINE: the same prompt via the population driver, which first asks the LLM to write
  `environment_card.md` (21 KB) and `expert_reward_context.md` (8 KB) and injects those
  -> generation 0 of the live EUREKA v9 run scored **0.18 / 2.08 / 3.90 / 6.67, all 0/20**.

Both paths use `_environment_block()`, which injects `environment_card.md` +
`expert_reward_context.md` from the run's `context/` directory. The difference is *which files
those are*. So this control writes the **pilot's** context files into a fresh run's context
directory, lets the driver reuse them instead of generating its own, and leaves everything else
(prompt, code path, population size, temperature, training) identical.

Interpretation, fixed in advance
--------------------------------
* candidates reach the ~300 range -> the self-generated environment analysis is what degrades
  generation, and the fix is to stop letting it overwrite the stated facts;
* candidates stay at 0-7 -> the degradation is in the population driver's generation path
  itself (message shape, budget, per-candidate temperature), not the context, and running CREATE
  would inherit exactly the same defect.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = r"D:\Code\python\research\llm_env_310\Scripts\python.exe"
KEY_FILE = Path(r"D:\Code\python\research\DSapi.txt")
PILOT_CTX = REPO / "runs/env_007/terminal_rule_pilot/seed_0/context"
PREFIX = "eureka_v9_pilotctx"
RUN_ROOT = REPO / "runs/env_007"
CTX = RUN_ROOT / PREFIX / "seed_0" / "context"
LOG = RUN_ROOT / "eureka_v9_pilotctx.log"
ERR = RUN_ROOT / "eureka_v9_pilotctx.err.log"

CMD = [PY, "-m", "pipeline.run_eureka_population",
       "--config", "configs/env007_fragilecargo_eureka_v9.yaml",
       "--prefix", PREFIX,
       "--generations", "1", "--population-size", "8", "--elite-size", "1",
       "--children-per-parent", "1", "--total-timesteps", "1200000",
       "--eval-episodes", "20", "--parallel-candidates", "3", "--seed", "0",
       "--skip-context"]


def main() -> None:
    CTX.mkdir(parents=True, exist_ok=True)
    for name in ("environment_card.md", "expert_reward_context.md"):
        shutil.copyfile(PILOT_CTX / name, CTX / name)
    print(f"seeded the run context from the pilot path: {sorted(p.name for p in CTX.iterdir())}",
          flush=True)

    key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key.startswith("sk-"):
        raise SystemExit("key file invalid")
    env = dict(os.environ)
    env.update({"EUREKA_DEEPSEEK_API_KEY": key, "DEEPSEEK_API_KEY": key,
                "DEEPSEEK_THINKING": "disabled", "OMP_NUM_THREADS": "6",
                "MKL_NUM_THREADS": "6"})

    for attempt in range(1, 13):
        stamp = time.strftime("%H:%M:%S")
        print(f"\n===== attempt {attempt} at {stamp} =====", flush=True)
        cmd = CMD + (["--resume"] if attempt > 1 else [])
        with open(LOG, "a", encoding="utf-8") as out, open(ERR, "a", encoding="utf-8") as err:
            out.write(f"\n===== attempt {attempt} at {stamp} =====\n")
            out.flush()
            rc = subprocess.call(cmd, cwd=REPO, stdout=out, stderr=err, env=env)
        print(f"attempt {attempt} exit {rc}", flush=True)
        summary = RUN_ROOT / PREFIX / "seed_0" / "eureka_summary.md"
        if rc == 0 and summary.exists():
            print("CONTROL RUN COMPLETE", flush=True)
            return
        time.sleep(60)
    print("attempt budget exhausted", flush=True)


if __name__ == "__main__":
    main()
