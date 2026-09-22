"""Smoke-test the reflection stage's structure-block injection on a real agent call.

The generation stage was verified end to end (its recorded prompt contains the block, and
the round-1 reward implements all nine native terms). The reflection stage is the other
consumer, and rounds 2-10 depend on it: it reads `v9_structure_block.md` from the directory
of the environment card, and only when
`context.include_reward_structure_in_reflection` is true.

This runs one real reflection call whose only purpose is to produce the recorded prompt; the
returned code is discarded. It writes into a scratch directory under runs/env_007 and does
not touch the live CREATE run.

Usage:
    python tools/smoke_reflection_structure_block.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KEY_FILE = Path(r"D:\Code\python\research\DSapi.txt")
CONFIG = "configs/env007_fragilecargo_create_v9.yaml"
SCRATCH = "smoke_reflection_v9block/seed_0/iter_02/generation"

sys.path.insert(0, str(REPO))
from pipeline.common import load_config, write_text  # noqa: E402
from pipeline.run_reflection_agent import run_reflection_agent  # noqa: E402


def main() -> int:
    os.environ.setdefault("DEEPSEEK_THINKING", "disabled")
    key = KEY_FILE.read_text(encoding="utf-8").strip()
    os.environ["DEEPSEEK_API_KEY"] = key
    os.environ["EUREKA_DEEPSEEK_API_KEY"] = key

    cfg = load_config(CONFIG)
    run_root = Path(cfg["experiment"]["run_root"])

    live_card_dir = run_root / "fragilecargo_create_v9/seed_0/iter_01/generation"
    if not (live_card_dir / "environment_card.md").exists():
        raise SystemExit(f"live round-1 card not found under {live_card_dir}")

    scratch = run_root / SCRATCH
    prev_reward = scratch / "previous_reward.py"
    write_text(prev_reward, "def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):\n"
                            "    components = {'progress': 0.0}\n"
                            "    return 0.0, components\n")
    train_dir = run_root / "smoke_reflection_v9block/seed_0/iter_01/training"
    write_text(train_dir / "training_feedback.md",
               "# Training feedback\nscore=3.10\nterminated=0/20\ndock_entered=0.00\n")
    memory = run_root / "smoke_reflection_v9block/seed_0/memory/reward_memory.md"
    write_text(memory, "# Reward Memory\n\n| iter | skeleton | score | best | delta | len | key_signal | action |\n"
                       "|---:|---|---:|---:|---:|---:|---|---|\n"
                       "| 1 | progress_only | 3.10 | 3.10 | 0.00 | 400.00 | progress=0.007 | new_best |\n")

    before = set(p.name for p in scratch.glob("*")) if scratch.exists() else set()
    run_reflection_agent(
        config_path=CONFIG,
        previous_reward_path=str(prev_reward),
        best_reward_path=None,
        train_run_dir=str(train_dir),
        memory_path=str(memory),
        out_run_name=SCRATCH,
        reward_version="v2",
        environment_card_path=str(live_card_dir / "environment_card.md"),
        mock=False,
    )

    record = scratch / "prompt_records" / "agent_reflection.md"
    text = record.read_text(encoding="utf-8")
    checks = {
        "block heading (6.5)": "# 6.5." in text,
        "nine-term table": "terminal_failure" in text and "approach_cargo" in text,
        "precedence declaration": "优先级声明" in text,
        "raw facts still present": "# 6. 环境事实" in text,
    }
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"recorded prompt: {record} ({len(text)} chars)")
    print(f"new files in scratch: {sorted(set(p.name for p in scratch.glob('*')) - before)}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
