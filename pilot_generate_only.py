"""Generate reward candidates with a given prompt, and score them mechanically.

No training. The point is that the structural checks in
`analyze_terminal_dominance.py` are computed in seconds on a quantity with a
dynamic range of ~1000x, so a prompt variant can be compared on 16 samples in
about the time a single training run takes to produce one noisy bit.

Usage
-----
    python pilot_generate_only.py \
        --config configs/env007_terminal_rule_pilot.yaml \
        --context runs/env_007/terminal_rule_pilot/seed_0/context \
        --prompt prompts/eureka_01_initial_reward_v3.md \
        --out runs/env_007/prompt_ab/v3 --n 16 --temperature 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.common import load_config  # noqa: E402
from pipeline.run_eureka_population import _make_client, materialise_reward  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--context", required=True, help="dir with environment_card.md")
    ap.add_argument("--prompt", required=True, help="initial-reward prompt file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--start-index", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["prompts"]["eureka_initial"] = args.prompt

    context = Path(args.context)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.json"
    manifest = {
        "config": args.config,
        "context": str(context),
        "prompt": args.prompt,
        "temperature": args.temperature,
        "model": cfg["llm"]["model_reward"],
        "candidates": [],
    }

    client = _make_client(cfg)
    print(f"prompt   : {args.prompt}")
    print(f"out      : {out_root}")
    print(f"model    : {cfg['llm']['model_reward']}  temperature={args.temperature}")

    for i in range(args.start_index, args.start_index + args.n):
        name = f"cand_{i:02d}"
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            reward_path = materialise_reward(
                cfg, client, context, out_dir, "initial", temperature=args.temperature)
            validation = json.loads(
                (out_dir / "validations" / "reward_v1.validation.json").read_text(encoding="utf-8"))
            ok = bool(validation.get("valid"))
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            manifest["candidates"].append({"name": name, "ok": False,
                                           "error": f"{type(exc).__name__}: {exc}"})
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            continue

        manifest["candidates"].append({
            "name": name,
            "ok": ok,
            "reward": str(reward_path),
            "errors": validation.get("errors", []),
            "seconds": round(time.time() - t0, 1),
        })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"  [{name}] valid={ok}  {time.time() - t0:.0f}s"
              f"{'' if ok else '  errors=' + str(validation.get('errors'))}", flush=True)


if __name__ == "__main__":
    main()
