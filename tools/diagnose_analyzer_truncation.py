"""Diagnose the truncated environment card before spending 5 h on the run.

The first CREATE attempt at 2026-09-22 02:19 wrote a 146-byte environment card:
the analyzer's completion stopped mid-sentence after ~50 characters, and
`DeepSeekClient.completion` only retries when the response is *empty*, so the truncated
text was written to disk and handed to the reward generator. Normal cards are 15-20 KB.

This script calls the analyzer in isolation (no files are written into the run tree) and
reports, per attempt, the content length, the finish_reason and whether the domain
geometry made it into the card. Enough good responses => transient, relaunch. All bad =>
systematic, and the pipeline needs a real truncation guard before the run is worth
restarting.

Usage:
    python tools/diagnose_analyzer_truncation.py [attempts]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
KEY_FILE = Path(r"D:\Code\python\research\DSapi.txt")
CONFIG = "configs/env007_fragilecargo_create_v9.yaml"

sys.path.insert(0, str(REPO))
from pipeline.common import load_config, read_text  # noqa: E402
from llm_clients.deepseek_client import DeepSeekClient  # noqa: E402

GEOMETRY_MARKERS = ("0.024", "0.030", "0.84", "0.60")


def main() -> int:
    attempts = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    os.environ.setdefault("DEEPSEEK_THINKING", "disabled")
    key = KEY_FILE.read_text(encoding="utf-8").strip()
    os.environ["DEEPSEEK_API_KEY"] = key
    os.environ["EUREKA_DEEPSEEK_API_KEY"] = key  # this config's llm.api_key_env (inherited)

    cfg = load_config(CONFIG)
    system_prompt = read_text(cfg["prompts"]["environment_analyzer"])
    task_spec = read_text(cfg["inputs"]["task_spec_path"])
    masked_step = read_text(cfg["inputs"]["masked_step_path"])
    user_prompt = f"ANONYMIZED_TASK_SPEC:\n{task_spec}\n\nMASKED_STEP_SOURCE:\n{masked_step}"

    llm = cfg["llm"]
    client = DeepSeekClient(api_key_env=llm["api_key_env"], base_url=llm["base_url"])
    min_chars = int(llm.get("min_chars_env_card", 8000))

    ok = 0
    for i in range(1, attempts + 1):
        try:
            response = client.completion(
                model=llm["model_env"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=llm["temperature_environment_analyzer"],
                max_tokens=llm["max_tokens_env"],
                min_content_chars=min_chars,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic, report anything
            print(f"attempt {i}: EXCEPTION {type(exc).__name__}: {exc}", flush=True)
            continue

        choice = response.choices[0]
        text = choice.message.content or ""
        has_geometry = any(marker in text for marker in GEOMETRY_MARKERS)
        complete = text.rstrip().endswith(("```", "。", "|", ")")) or len(text) > 8000
        status = "OK" if (len(text) > 8000 and has_geometry) else "BAD"
        if status == "OK":
            ok += 1
        print(
            f"attempt {i}: {status} chars={len(text)} finish_reason={choice.finish_reason!r} "
            f"geometry={has_geometry} ends_clean={complete}",
            flush=True,
        )

    print(f"\n{ok}/{attempts} usable responses")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
