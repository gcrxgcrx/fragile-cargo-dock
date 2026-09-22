"""End-to-end verification of the context fix (see PIPELINE_CONTEXT_DIAGNOSIS.md).

Runs the EUREKA driver with `configs/env007_fragilecargo_eureka_v9_fixedspec.yaml`, whose only
difference from the config used by the failed run is `inputs.task_spec_path` -> the v2 spec that
carries the dock geometry. If the diagnosis is right, the generated candidates should now use
`|obs[12]| <= 0.024 and |obs[13]| <= 0.030` as their "inside the dock" test (the failed run's
eight candidates either omitted it or guessed 0.20-0.25), and at least one should score in the
delivery range rather than 0-7.

Two candidates only (`population_size` is forced to `elite*(1+children)` = 2 by the driver), at
1.2 M steps, evaluated afterwards on the fresh block 39000-39059 - the same block used for the
pilot-context control, so all three arms are comparable.
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
PREFIX = "eureka_v9_fixedspec"
CFG = "configs/env007_fragilecargo_eureka_v9_fixedspec.yaml"
RUN = REPO / "runs/env_007" / PREFIX
LOG = RUN.parent / f"{PREFIX}.log"
ERR = RUN.parent / f"{PREFIX}.err.log"


def main() -> None:
    # the analyzer card was generated into runs/env_007/<prefix>/environment_card.md; the driver
    # expects it at runs/env_007/<prefix>/seed_0/context/environment_card.md. Move it and let the
    # driver build the expert context from the same (fixed) spec.
    ctx = RUN / "seed_0" / "context"
    ctx.mkdir(parents=True, exist_ok=True)
    src = RUN / "environment_card.md"
    if src.exists():
        (ctx / "environment_card.md").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"moved the regenerated card into {ctx}", flush=True)
    if not (ctx / "environment_card.md").exists():
        raise SystemExit("no environment card; run the analyzer first")

    key = KEY_FILE.read_text(encoding="utf-8").strip()
    env = dict(os.environ)
    env.update({"EUREKA_DEEPSEEK_API_KEY": key, "DEEPSEEK_API_KEY": key,
                "DEEPSEEK_THINKING": "disabled", "OMP_NUM_THREADS": "6",
                "MKL_NUM_THREADS": "6"})

    cmd = [PY, "-m", "pipeline.run_eureka_population", "--config", CFG, "--prefix", PREFIX,
           "--generations", "1", "--population-size", "2", "--elite-size", "1",
           "--children-per-parent", "1", "--total-timesteps", "1200000",
           "--eval-episodes", "20", "--parallel-candidates", "2", "--seed", "0"]
    for attempt in range(1, 7):
        stamp = time.strftime("%H:%M:%S")
        print(f"===== attempt {attempt} at {stamp} =====", flush=True)
        with open(LOG, "a", encoding="utf-8") as out, open(ERR, "a", encoding="utf-8") as err:
            rc = subprocess.call(cmd + (["--resume"] if attempt > 1 else []),
                                 cwd=REPO, stdout=out, stderr=err, env=env)
        summary = RUN / "seed_0" / "eureka_summary.md"
        if rc == 0 and summary.exists():
            print("RUN COMPLETE", flush=True)
            break
        print(f"attempt {attempt} exit {rc}; retrying", flush=True)
        time.sleep(60)

    # report: tolerance used by each candidate + its score
    print(f"\n{'candidate':<16} {'uses 0.024/0.030':>17} {'score':>9} {'terminated':>11}",
          flush=True)
    dirs = []
    for rp in sorted(RUN.glob("seed_0/gen_*/cand_*/reward_v1.py")):
        txt = rp.read_text(encoding="utf-8", errors="ignore")
        ok = ("0.024" in txt) and ("0.030" in txt)
        sp = rp.parent / "training" / "training_summary.json"
        sc = term = None
        if sp.exists():
            d = json.loads(sp.read_text(encoding="utf-8-sig"))
            e = d.get("external_eval") or {}
            sc, term = e.get("mean_eval_reward"), e.get("termination_breakdown", {}).get("terminated")
            dirs.append(str(sp.parent.relative_to(REPO)).replace("\\", "/"))
        print(f"{str(rp.parent.relative_to(RUN / 'seed_0')):<16} {str(ok):>17} "
              f"{(round(sc, 3) if isinstance(sc, float) else sc)!s:>9} {str(term) + '/20':>11}",
              flush=True)

    if dirs:
        print(f"\nscoring {len(dirs)} candidates on fresh block 39000-39059", flush=True)
        subprocess.run([PY, "eval_pool.py", "--episodes", "60", "--seed-offset", "39000",
                        "--out", str(RUN / "eval_block39000.json"), *dirs], cwd=REPO, check=False)
        for r in json.loads((RUN / "eval_block39000.json").read_text(encoding="utf-8-sig")):
            print(f"  {r['run_dir']:<58} success {r['success']}/60  dock {r['dock_rate']:.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
