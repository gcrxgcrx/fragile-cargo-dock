"""Regression check: `run_investigator` must accept the config's `max_turns`.

Every reflection call passes `max_turns=subagent_investigator.max_turns` from the config.
`run_investigator` did not accept it, so it raised TypeError, the caller caught it and
continued without a research signal, and the whole subagent stage was silently skipped in
every round of `fragilecargo_create_v2` (10 rounds, 10 skipped calls).

Uses a fake client: no API call, no key needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.subagent_investigator import run_investigator  # noqa: E402

TRAIN_DIRS = [
    REPO / "runs/env_007/fragilecargo_create_v9/seed_0/iter_01/training",
    REPO / "runs/env_007/fragilecargo_create_v2/seed_0/iter_01/training",
]

PAYLOAD = (
    '{"key_findings":"score 4.50, 0/20 terminated",'
    '"component_anomalies":"crate_dock_alignment 81% of magnitude",'
    '"training_dynamics":"flat across checkpoints",'
    '"signal_quality":"completion predicate never reached",'
    '"confidence":"medium"}'
)


class _Message:
    content = PAYLOAD


class _Choice:
    message = _Message()


class _Response:
    choices = [_Choice()]


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def completion(self, **kwargs):
        # The investigator must not forward max_turns to the API client.
        assert "max_turns" not in kwargs, "max_turns leaked into the API call"
        self.calls += 1
        return _Response()


def main() -> int:
    train_dir = next((d for d in TRAIN_DIRS if d.exists()), None)
    if train_dir is None:
        raise SystemExit(f"no training dir found among {[str(d) for d in TRAIN_DIRS]}")

    client = FakeClient()
    result = run_investigator(
        train_dir=str(train_dir),
        previous_reward_path="",
        memory_path="",
        client=client,
        model="deepseek-flash",
        max_turns=3,  # exactly what run_reflection_agent.py passes
    )
    errors = [t for t in result["tool_trace"] if "error" in t]
    print(f"train_dir        : {train_dir}")
    print(f"turns_used       : {result['turns_used']}")
    print(f"signal produced  : {bool(result['research_signal_text'])}")
    print(f"tool_trace errors: {errors}")
    if errors:
        print("FAIL: investigator still errors")
        return 1
    print("PASS: max_turns accepted, signal produced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
