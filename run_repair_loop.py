"""Drive the evidence-channel repair experiment (runs/env_007/REPAIR_LOOP_PREREGISTRATION.md).

Two arms, identical in everything except the *content* of the component-level evidence handed
to the model:

    real : build_reward_reflection(<target's 1.2M training dir>)      # the pipeline's own report
    sham : the same report with every per-component table's rows shuffled and each value
           column shuffled independently (shape and marginals preserved, name<->value
           correspondence and magnitude ordering destroyed)

Operator: `materialise_reward(mode="edit")` from `pipeline/run_eureka_population.py`, i.e. the
same code path, prompt and validation the project's own edit flow uses. Nothing here is a new
channel: the point of the experiment is that this channel *already exists* (SESSION_STATE §3g:
EUREKA's reflection already carries per-component active_rate, despite the function docstring
claiming otherwise).

Usage
-----
    python run_repair_loop.py --arm real --dry-run
    python run_repair_loop.py --arm sham --dry-run
    python run_repair_loop.py --arm real          # needs EUREKA_DEEPSEEK_API_KEY + DEEPSEEK_THINKING=disabled
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.common import load_config
from pipeline.run_eureka_population import (
    _make_client, build_reward_reflection, materialise_reward,
)

# target name -> (reward file, 1.2M training dir)
TARGETS = {
    "L2_cand_00": ("runs/env_007/prompt_ladder/L2/cand_00/reward_v1.py",
                   "runs/env_007/ladder_train/L2_cand_00"),
    "L2_cand_05": ("runs/env_007/prompt_ladder/L2/cand_05/reward_v1.py",
                   "runs/env_007/ladder_train/L2_cand_05"),
    "L2_cand_04": ("runs/env_007/prompt_ladder/L2/cand_04/reward_v1.py",
                   "runs/env_007/ladder_train/L2_cand_04"),
    "L2_cand_14": ("runs/env_007/prompt_ladder/L2/cand_14/reward_v1.py",
                   "runs/env_007/ladder_train/L2_cand_14"),
    "L0_cand_02": ("runs/env_007/prompt_ladder/L0/cand_02/reward_v1.py",
                   "runs/env_007/ladder_train/L0_cand_02"),
    "L0_cand_11": ("runs/env_007/prompt_ladder/L0/cand_11/reward_v1.py",
                   "runs/env_007/ladder_train/L0_cand_11"),
    "L0_cand_13": ("runs/env_007/prompt_ladder/L0/cand_13/reward_v1.py",
                   "runs/env_007/ladder_train/L0_cand_13"),
    "L0_cand_00": ("runs/env_007/prompt_ladder/L0/cand_00/reward_v1.py",
                   "runs/env_007/ladder_train/L0_cand_00"),
}


def shamify(reflection: str, seed: int) -> tuple[str, int]:
    """Shuffle rows and then each value column, inside every per-component table."""
    rng = random.Random(seed)
    lines = reflection.split("\n")
    out: list[str] = []
    tables = 0
    i = 0
    while i < len(lines):
        if lines[i].startswith("| component |"):
            header = lines[i]
            sep = lines[i + 1] if i + 1 < len(lines) else ""
            j = i + 2
            rows = []
            while j < len(lines) and lines[j].startswith("|"):
                rows.append(lines[j])
                j += 1
            parsed = [[c.strip() for c in r.split("|")[1:-1]] for r in rows]
            ncols = len(parsed[0]) if parsed else 0
            rng.shuffle(parsed)                      # kill the magnitude ordering
            for c in range(1, ncols):                # kill name <-> value correspondence
                col = [row[c] for row in parsed]
                rng.shuffle(col)
                for k, row in enumerate(parsed):
                    row[c] = col[k]
            out += [header, sep] + ["| " + " | ".join(row) + " |" for row in parsed]
            tables += 1
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out), tables


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["real", "sham"], required=True)
    ap.add_argument("--config", default="configs/env007_terminal_rule_pilot.yaml")
    ap.add_argument("--context", default="runs/env_007/terminal_rule_pilot/seed_0/context")
    ap.add_argument("--out-root", default="runs/env_007/repair_loop")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0, help="sham shuffle seed")
    ap.add_argument("--only", default=None, help="comma-separated target subset")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    context = Path(args.context)
    out_root = Path(args.out_root) / args.arm
    out_root.mkdir(parents=True, exist_ok=True)

    targets = list(TARGETS)
    if args.only:
        wanted = set(args.only.split(","))
        targets = [t for t in targets if t in wanted]

    if not args.dry_run:
        client = _make_client(cfg)
    else:
        client = None

    manifest = {
        "arm": args.arm, "config": args.config, "context": str(context),
        "edit_prompt": cfg["prompts"].get("eureka_edit"),
        "model": cfg["llm"]["model_reward"], "temperature": args.temperature,
        "dry_run": bool(args.dry_run), "sham_seed": args.seed, "targets": [],
    }

    for name in targets:
        reward_rel, train_rel = TARGETS[name]
        parent_path = Path(reward_rel)
        train_dir = Path(train_rel)
        parent_code = parent_path.read_text(encoding="utf-8")

        reflection = build_reward_reflection(train_dir)
        summary = json.loads((train_dir / "training_summary.json").read_text(encoding="utf-8"))
        parent_score = float(summary.get("external_eval", {}).get("mean_eval_reward", 0.0))

        tables = 0
        if args.arm == "sham":
            reflection, tables = shamify(reflection, args.seed)

        entry = {"target": name, "parent": reward_rel, "train_dir": train_rel,
                 "parent_score": round(parent_score, 4),
                 "reflection_chars": len(reflection), "component_tables_shuffled": tables,
                 "has_active_rate_table": "activation rate" in reflection}

        if args.dry_run:
            d = out_root / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "reflection_used.md").write_text(reflection, encoding="utf-8")
            entry["status"] = "dry-run"
            print(f"  {name:<12} score={parent_score:>8.3f}  refl={len(reflection):>6} chars  "
                  f"tables_shuffled={tables}  active_rate_table={entry['has_active_rate_table']}")
            manifest["targets"].append(entry)
            continue

        d = out_root / name / "cand_00"
        d.mkdir(parents=True, exist_ok=True)
        (out_root / name / "reflection_used.md").write_text(reflection, encoding="utf-8")
        try:
            path = materialise_reward(cfg, client, context, d, "edit",
                                      parent_code=parent_code, reflection=reflection,
                                      parent_score=parent_score,
                                      temperature=args.temperature)
            valid = json.loads((d / "validations" / "reward_v1.validation.json")
                               .read_text(encoding="utf-8")).get("valid", False)
            entry.update({"status": "ok" if valid else "invalid", "reward": str(path)})
            print(f"  {name:<12} -> {path}  valid={valid}", flush=True)
        except Exception as exc:  # noqa: BLE001
            entry.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {name:<12} -> ERROR {type(exc).__name__}: {exc}", flush=True)

        manifest["targets"].append(entry)
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ok = sum(1 for t in manifest["targets"] if t.get("status") == "ok")
    print(f"\narm={args.arm}: {len(manifest['targets'])} targets, {ok} valid rewards, "
          f"wrote {out_root/'manifest.json'}")


if __name__ == "__main__":
    main()
