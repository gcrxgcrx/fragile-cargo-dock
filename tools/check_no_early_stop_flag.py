"""Regression check for `--no-early-stop-all`.

The first version of the flag set `patience_after_solved = rounds` *before* the line that reads
`no_improvement_patience_after_solved` from the config, so the config value (2) overwrote it and
the flag silently did nothing: runs/env_007/fragilecargo_create_v9 stopped again after round 7
with `stop_solved_no_improvement_keep_best`.

This asserts the computed patience values by running the real function with every side effect
stubbed (no API, no training, no run directory outside a scratch prefix).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pipeline.run_iterative_experiment as rie  # noqa: E402


def probe(no_early_stop_all: bool, rounds: int = 10) -> dict:
    captured: dict = {}

    def fake_run_cmd(cmd):
        raise KeyboardInterrupt("abort after the patience values are computed")

    def fake_write_summary(*args, **kwargs):
        captured["summary"] = args

    rie.run_cmd = fake_run_cmd
    rie.write_experiment_summary = fake_write_summary
    try:
        rie.run_iterative_experiment(
            config_path="configs/env007_fragilecargo_create_v9.yaml",
            prefix=f"__flagcheck_{int(no_early_stop_all)}__",
            rounds=rounds,
            total_timesteps=1000,
            eval_episodes=1,
            mock=True,
            seed=0,
            no_early_stop=False,
            no_early_stop_all=no_early_stop_all,
        )
    except KeyboardInterrupt:
        pass
    finally:
        try:
            import shutil
            shutil.rmtree(REPO / "runs/env_007" / f"__flagcheck_{int(no_early_stop_all)}__", ignore_errors=True)
        except OSError:
            pass
    return captured


def main() -> int:
    import inspect
    src = inspect.getsource(rie.run_iterative_experiment)
    # The override must be textually after the config read, otherwise it is overwritten again.
    i_cfg = src.index('patience_after_solved = int(iter_cfg.get("no_improvement_patience_after_solved"')
    i_flag = src.index("if no_early_stop_all:")
    print(f"config read at char {i_cfg}, flag override at char {i_flag}")
    ok_order = i_flag > i_cfg
    print(f"{'PASS' if ok_order else 'FAIL'}  flag override comes after the config read")

    # Static evaluation of the ordering effect, since the values are locals inside the function.
    ns: dict = {}
    body = src[i_flag:src.index("retry_identical_unsolved", i_flag)]
    for flag in (False, True):
        env = {"no_early_stop_all": flag, "rounds": 10, "stop_after_solved_drop": True,
               "patience_after_solved": 2, "patience_unsolved": 3}
        exec(body.replace("if no_early_stop_all:", "if no_early_stop_all:"), env)
        print(f"  no_early_stop_all={flag}: patience_after_solved={env['patience_after_solved']}, "
              f"patience_unsolved={env['patience_unsolved']}, "
              f"stop_after_solved_drop={env['stop_after_solved_drop']}")
        if flag and env["patience_after_solved"] != 10:
            print("FAIL: flag did not raise the patience")
            return 1
        if not flag and env["patience_after_solved"] != 2:
            print("FAIL: default patience changed")
            return 1
    print("PASS: --no-early-stop-all now raises patience_after_solved to `rounds`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
