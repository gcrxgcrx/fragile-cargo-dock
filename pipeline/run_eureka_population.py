"""EUREKA-style population reward search (baseline M3).

This is the comparison baseline for CREATE, not a CREATE variant. It follows
EUREKA's reward-evolution recipe rather than CREATE's single-lineage,
structured-diagnosis loop:

    generation 0 : sample ``population_size`` independent reward candidates
    generation g : rank by native task score, keep the top ``elite_size``,
                   each elite spawns ``children_per_parent`` LLM edits, the
                   next population is elites + children, repeat

Differences from ``pipeline.run_iterative_experiment`` that matter for a fair
comparison:

* **No structured diagnosis.** The only feedback handed to the LLM is the
  *reward reflection*: the task score plus the evolution of the reward's own
  component values over training. There is no signal-completeness audit, no
  active-rate / magnitude-share table, no causal-chain record, no diagnostic
  subagent, no Level 1/2/3 intervention taxonomy, no cross-round memory.
* **Population, not lineage.** Several candidates are trained per generation
  and selection is by native score; CREATE refines one lineage.
* **Single training seed per candidate**, as in EUREKA's search stage.

Every candidate is trained through ``training.train_sb3_wrapper`` so that the
artifacts (training_summary.json, training_feedback.md, component_stats.md,
monitor CSVs) are identical in kind to the ones CREATE produces, and the score
used for selection is the same native objective:

    J = training_summary.json -> external_eval.mean_eval_reward

The flow is resumable: state is checkpointed to ``population.json`` after every
candidate, and ``--resume`` skips anything already scored.

``--mock`` runs the whole pipeline without any LLM call (fixed template reward).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import load_config, read_text, write_text, write_json, record_prompt, record_response
from .run_03_direct_reward_generator import extract_code, validate_code

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def run_cmd(cmd: List[str]) -> None:
    cmd = [sys.executable if arg == "python" else arg for arg in cmd]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _pad(n: int) -> str:
    return f"{n:02d}"


MOCK_REWARD_MD = """```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Mock reward used only for plumbing tests (--mock). It rewards closing the
    # distance between the payload and its goal, with a light action cost.
    to_goal_x = next_obs[12]
    to_goal_y = next_obs[13]
    old_x = obs[12]
    old_y = obs[13]

    old_distance = (old_x * old_x + old_y * old_y) ** 0.5
    new_distance = (to_goal_x * to_goal_x + to_goal_y * to_goal_y) ** 0.5

    progress_reward = 5.0 * (old_distance - new_distance)
    action_cost = -0.01 * (action[0] * action[0] + action[1] * action[1])

    total_reward = progress_reward + action_cost
    components = {
        "progress_reward": progress_reward,
        "action_cost": action_cost,
        "total_reward": total_reward,
    }
    if isinstance(info, dict):
        info["reward_terms"] = components
    return float(total_reward), components
```
"""


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

def experiment_root(cfg, prefix: str, seed: int) -> Path:
    return Path(cfg["experiment"]["run_root"]) / prefix / f"seed_{seed}"


def context_dir(cfg, prefix: str, seed: int) -> Path:
    return experiment_root(cfg, prefix, seed) / "context"


def candidate_dir(cfg, prefix: str, seed: int, gen: int, idx: int) -> Path:
    return experiment_root(cfg, prefix, seed) / f"gen_{_pad(gen)}" / f"cand_{_pad(idx)}"


# --------------------------------------------------------------------------- #
# Scoring / reflection
# --------------------------------------------------------------------------- #

def read_native_score(train_dir: Path) -> Optional[float]:
    """J = external_eval.mean_eval_reward, the environment's own objective."""
    summary = Path(train_dir) / "training_summary.json"
    if not summary.exists():
        return None
    data = json.loads(summary.read_text(encoding="utf-8"))
    value = data.get("external_eval", {}).get("mean_eval_reward")
    return float(value) if value is not None else None


def _read_monitor_curve(train_dir: Path, n_points: int = 6) -> List[Dict[str, float]]:
    """Episode return sampled at ``n_points`` quantiles of the training run.

    EUREKA's reward reflection includes how the reward evolved during training.
    ``train_sb3_wrapper`` writes one SB3 Monitor CSV per vectorised worker; this
    reconstructs a single ordered curve from all of them.
    """
    monitor_dir = Path(train_dir) / "monitor"
    if not monitor_dir.is_dir():
        return []

    rows: List[Dict[str, float]] = []
    for csv_path in sorted(monitor_dir.glob("*.csv")):
        header = None
        for raw in csv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not raw.strip() or raw.startswith("#"):
                continue
            parts = [p.strip() for p in raw.split(",")]
            if header is None:
                header = parts
                continue
            record = dict(zip(header, parts))
            try:
                rows.append({
                    "r": float(record.get("r", 0.0)),
                    "l": float(record.get("l", 0.0)),
                    "generated": float(record.get("generated_reward", "nan") or "nan"),
                    "original": float(record.get("original_env_reward", "nan") or "nan"),
                })
            except (TypeError, ValueError):
                continue

    if not rows:
        return []

    out = []
    for i in range(n_points):
        lo = int(len(rows) * i / n_points)
        hi = int(len(rows) * (i + 1) / n_points)
        chunk = rows[lo:hi] or rows[lo:lo + 1]
        out.append({
            "fraction": (i + 1) / n_points,
            "mean_return": statistics.mean(c["r"] for c in chunk),
            "mean_length": statistics.mean(c["l"] for c in chunk),
        })
    return out


def build_reward_reflection(train_dir: Path) -> str:
    """EUREKA-style reward reflection: task score + the reward's own components.

    Deliberately *not* a structured diagnosis: no active-rate / share / signed
    decomposition, no behaviour or termination interpretation.
    """
    summary_path = Path(train_dir) / "training_summary.json"
    if not summary_path.exists():
        return "(no training summary available)"

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    external = data.get("external_eval", {})
    comp = data.get("component_summary", {}).get("episode_component_sum_stats", {})

    lines = ["### Task score", ""]
    lines.append(f"- mean_eval_reward: {external.get('mean_eval_reward', '?')}")
    lines.append(f"- mean_episode_length: {external.get('mean_episode_length', '?')}")
    lines.append(f"- eval episodes: {external.get('eval_episodes', '?')}")
    term = external.get("termination_breakdown", {})
    if term:
        lines.append(f"- termination breakdown: {term}")

    curve = _read_monitor_curve(train_dir)
    if curve:
        lines += ["", "### Episode return during training", "",
                  "| training progress | mean episode return | mean episode length |",
                  "|---|---:|---:|"]
        for point in curve:
            lines.append(f"| {point['fraction']:.0%} | {point['mean_return']:.2f} | "
                         f"{point['mean_length']:.1f} |")

    # Per-component behaviour over successive windows of training. This is the
    # evidence that exposes degenerate rewards, e.g. a component that never
    # fires because the policy found a way to satisfy the reward without doing
    # the task. Written by RewardComponentStatsCallback. Exposed at the top
    # level of training_summary.json; older files only had it under
    # component_summary.
    snapshots = (data.get("monitor_snapshots")
                 or data.get("component_summary", {}).get("monitor_snapshots")
                 or [])
    if snapshots:
        total_steps = max(1, snapshots[-1].get("steps", 1))
        headers = [f"{int(round(s.get('steps', 0) / total_steps * 100))}%" for s in snapshots]

        names, final_magnitude = [], {}
        for snap in snapshots:
            for entry in snap.get("top_components", []):
                if entry["name"] not in final_magnitude:
                    names.append(entry["name"])
                final_magnitude[entry["name"]] = abs(entry.get("episode_sum_mean", 0.0))
        names.sort(key=lambda n: final_magnitude.get(n, 0.0), reverse=True)
        names = names[:12]

        def table(getter, title, fmt):
            rows = [f"| component | {' | '.join(headers)} |", "|---|" + "---:|" * len(headers)]
            for name in names:
                cells = []
                for snap in snapshots:
                    value = None
                    for entry in snap.get("top_components", []):
                        if entry["name"] == name:
                            value = getter(entry)
                            break
                    cells.append(fmt(value) if value is not None else "—")
                rows.append(f"| {name} | {' | '.join(cells)} |")
            return ["", f"### {title}", ""] + rows

        lines += table(lambda c: c.get("episode_sum_mean"),
                       "Reward component values during training (mean reward per episode)",
                       lambda v: f"{v:.3f}")
        lines += table(lambda c: c.get("active_rate"),
                       "Reward component activation rate during training "
                       "(fraction of steps where the component is non-zero)",
                       lambda v: f"{v:.1%}")

        if len(snapshots) == 1:
            lines += ["", "_Only one checkpoint was recorded; the training run was "
                          "shorter than the snapshot interval._"]

    if comp:
        lines += ["", "### Reward component values (episode sums over all training episodes)", "",
                  "| component | mean | abs mean | min | max |", "|---|---:|---:|---:|---:|"]
        for name in sorted(comp):
            item = comp[name]
            lines.append(
                f"| {name} | {item.get('mean', 0):.4f} | {item.get('abs_mean', 0):.4f} | "
                f"{item.get('min', 0):.4f} | {item.get('max', 0):.4f} |"
            )
    else:
        lines += ["", "### Reward component values", "", "(no component statistics available)"]

    errors = data.get("component_summary", {}).get("reward_error_count_max")
    if errors:
        lines += ["", f"- reward function errors during training: {errors}"]

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# LLM calls
# --------------------------------------------------------------------------- #

def _make_client(cfg):
    from llm_clients.deepseek_client import DeepSeekClient
    llm = cfg["llm"]
    return DeepSeekClient(api_key_env=llm["api_key_env"], base_url=llm["base_url"])


def _environment_block(context: Path) -> str:
    card = context / "environment_card.md"
    expert = context / "expert_reward_context.md"
    parts = ["# Environment card", read_text(card) if card.exists() else "(missing)"]
    if expert.exists():
        parts += ["", "# Expert reward context", read_text(expert)]
    return "\n\n".join(parts)


def generate_initial(cfg, client, context: Path, out_dir: Path, temperature: float) -> str:
    system_prompt = read_text(cfg["prompts"]["eureka_initial"])
    user_prompt = _environment_block(context)
    write_text(out_dir / "llm_inputs" / "initial.input.md", user_prompt)
    record_prompt(out_dir, "eureka_initial", system_prompt, user_prompt)

    if client is None:
        out_md = MOCK_REWARD_MD
    else:
        out_md = client.chat(
            model=cfg["llm"]["model_reward"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=cfg["llm"]["max_tokens_reward"],
            json_mode=False,
        )
    record_response(out_dir, "eureka_initial", out_md)
    return out_md


def generate_edit(cfg, client, context: Path, out_dir: Path, parent_code: str,
                  reflection: str, parent_score: float, temperature: float) -> str:
    system_prompt = read_text(cfg["prompts"]["eureka_edit"])
    user_prompt = "\n\n".join([
        _environment_block(context),
        "# Current reward function\n```python\n" + parent_code.strip() + "\n```",
        f"# Reward reflection of the current reward (native task score = {parent_score:.4f})\n"
        + reflection,
    ])
    write_text(out_dir / "llm_inputs" / "edit.input.md", user_prompt)
    record_prompt(out_dir, "eureka_edit", system_prompt, user_prompt)

    if client is None:
        out_md = MOCK_REWARD_MD
    else:
        out_md = client.chat(
            model=cfg["llm"]["model_reward"],
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=cfg["llm"]["max_tokens_reward"],
            json_mode=False,
        )
    record_response(out_dir, "eureka_edit", out_md)
    return out_md


def materialise_reward(cfg, client, context: Path, out_dir: Path, mode: str,
                       parent_code: str = "", reflection: str = "",
                       parent_score: float = 0.0,
                       temperature: float = 0.1) -> Path:
    """Ask the LLM for a reward, validate it, and retry with the errors attached."""
    max_retries = int(cfg.get("eureka", {}).get("max_validation_retries", 3))
    reward_path = out_dir / "reward_v1.py"
    last_errors: List[str] = []

    for attempt in range(max_retries + 1):
        if mode == "initial":
            out_md = generate_initial(cfg, client, context, out_dir, temperature)
        else:
            prompt_parent = parent_code
            prompt_reflection = reflection
            if attempt and last_errors:
                prompt_reflection += (
                    "\n\n### IMPORTANT: your previous draft failed validation\n"
                    + "\n".join(f"- {e}" for e in last_errors)
                )
            out_md = generate_edit(cfg, client, context, out_dir, prompt_parent,
                                   prompt_reflection, parent_score, temperature)

        code = extract_code(out_md)
        validation = validate_code(code)
        write_text(out_dir / f"draft_attempt{attempt}.md", out_md)
        write_json(out_dir / "validations" / f"draft_attempt{attempt}.json", validation)

        if validation["valid"]:
            write_text(out_dir / "reward_v1.md", out_md)
            write_text(reward_path, code)
            write_json(out_dir / "validations" / "reward_v1.validation.json", validation)
            return reward_path

        last_errors = validation["errors"]
        print(f"    validation failed (attempt {attempt + 1}/{max_retries + 1}): {last_errors}")

    write_text(out_dir / "reward_v1.md", out_md)
    write_text(reward_path, code)
    write_json(out_dir / "validations" / "reward_v1.validation.json",
               {"valid": False, "errors": last_errors, "warnings": []})
    return reward_path


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_candidate(config_path: str, cfg, prefix: str, seed: int, gen: int, idx: int,
                    reward_path: Path, total_timesteps: int, eval_episodes: int) -> Path:
    train_dir = candidate_dir(cfg, prefix, seed, gen, idx) / "training"
    train_dir.mkdir(parents=True, exist_ok=True)
    run_cmd([
        "python", "-m", "training.train_sb3_wrapper",
        "--config", config_path,
        "--reward", str(reward_path),
        "--run-name", f"{prefix}/seed_{seed}/gen_{_pad(gen)}/cand_{_pad(idx)}/training",
        "--save-dir", str(train_dir),
        "--total-timesteps", str(total_timesteps),
        "--eval-episodes", str(eval_episodes),
        "--seed", str(seed),
    ])
    return train_dir


# --------------------------------------------------------------------------- #
# Population state
# --------------------------------------------------------------------------- #

def load_state(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"generations": [], "best": None}


def save_state(path: Path, state: Dict[str, Any]) -> None:
    write_json(path, state)


def already_done(state: Dict[str, Any], gen: int, idx: int) -> Optional[Dict[str, Any]]:
    for block in state["generations"]:
        if block["gen"] != gen:
            continue
        for cand in block["candidates"]:
            if cand["idx"] == idx and cand.get("score") is not None:
                return cand
    return None


def record_candidate(state: Dict[str, Any], gen: int, cand: Dict[str, Any]) -> None:
    for block in state["generations"]:
        if block["gen"] == gen:
            block["candidates"] = [c for c in block["candidates"] if c["idx"] != cand["idx"]]
            block["candidates"].append(cand)
            block["candidates"].sort(key=lambda c: c["idx"])
            return
    state["generations"].append({"gen": gen, "candidates": [cand]})
    state["generations"].sort(key=lambda b: b["gen"])


def ranked(state: Dict[str, Any], gen: int) -> List[Dict[str, Any]]:
    for block in state["generations"]:
        if block["gen"] == gen:
            scored = [c for c in block["candidates"] if c.get("score") is not None]
            return sorted(scored, key=lambda c: c["score"], reverse=True)
    return []


def update_best(state: Dict[str, Any], cand: Dict[str, Any]) -> None:
    best = state.get("best")
    if cand.get("score") is None:
        return
    if best is None or cand["score"] > best["score"]:
        state["best"] = {
            "id": cand["id"],
            "score": cand["score"],
            "reward_path": cand["reward_path"],
        }


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def run_eureka_population(config_path: str, prefix: Optional[str] = None,
                          generations: Optional[int] = None,
                          population_size: Optional[int] = None,
                          elite_size: Optional[int] = None,
                          children_per_parent: Optional[int] = None,
                          total_timesteps: Optional[int] = None,
                          eval_episodes: Optional[int] = None,
                          parallel: Optional[int] = None,
                          seed: int = 0, mock: bool = False,
                          resume: bool = False, skip_context: bool = False) -> None:
    cfg = load_config(config_path)
    eureka = cfg.get("eureka", {})
    iteration = cfg.get("iteration", {})
    training = cfg.get("training", {})
    llm = cfg["llm"]

    prefix = prefix or iteration.get("experiment_prefix", "eureka")
    generations = int(generations or eureka.get("generations", 4))
    population_size = int(population_size or eureka.get("population_size", 4))
    elite_size = int(elite_size or eureka.get("elite_size", 2))
    children_per_parent = int(children_per_parent or eureka.get("children_per_parent", 1))
    total_timesteps = int(total_timesteps or training.get("total_timesteps", 1_000_000))
    eval_episodes = int(eval_episodes or training.get("eval_episodes", 20))
    parallel = int(parallel or eureka.get("parallel_candidates", 1))

    expected = elite_size * (1 + children_per_parent)
    if population_size != expected:
        print(f"NOTE: population_size {population_size} != elite*(1+children) = {expected}; "
              f"using {expected} candidates after generation 0")
        population_size = expected

    root = experiment_root(cfg, prefix, seed)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "population.json"
    state = load_state(state_path) if resume else {"generations": [], "best": None}

    print("=" * 70)
    print(" EUREKA-style population reward search (baseline M3)")
    print("=" * 70)
    print(f" config           : {config_path}")
    print(f" env              : {training.get('runner_env_id')}")
    print(f" generations      : {generations}")
    print(f" population       : {population_size} (elite {elite_size} + "
          f"{elite_size}x{children_per_parent} children)")
    print(f" timesteps/cand   : {total_timesteps:,}")
    print(f" eval episodes    : {eval_episodes}")
    print(f" concurrent runs  : {parallel} (candidates within a generation are independent)")
    print(f" selection metric : external_eval.mean_eval_reward (native)")
    print(f" seed             : {seed}")
    print(f" output           : {root}")
    print(f" api key env var  : {llm['api_key_env']}")
    print("=" * 70, flush=True)

    # ---- shared environment context ---------------------------------------
    ctx = context_dir(cfg, prefix, seed)
    ctx.mkdir(parents=True, exist_ok=True)
    if not skip_context and not (ctx / "environment_card.md").exists():
        print("\n[context] analysing the environment (shared by all candidates)")
        from .run_01_environment_analyzer_md import run as run_env
        from .run_02_build_expert_context import run as run_expert
        ctx_run_name = str(ctx.relative_to(Path(cfg["experiment"]["run_root"])))
        run_env(config_path, ctx_run_name, mock=mock)
        run_expert(config_path, ctx_run_name)

    if not (ctx / "environment_card.md").exists():
        raise RuntimeError(
            f"missing {ctx / 'environment_card.md'}; run without --skip-context first")

    client = None if mock else _make_client(cfg)

    for gen in range(generations):
        print("\n" + "-" * 70)
        print(f" GENERATION {gen}")
        print("-" * 70, flush=True)

        if gen == 0:
            work = [{"slot": i, "parent": None} for i in range(population_size)]
        else:
            parents = ranked(state, gen - 1)[:elite_size]
            if not parents:
                raise RuntimeError(f"generation {gen - 1} produced no scored candidates")
            print(f" elites carried over: {[p['id'] for p in parents]}")

            # Elites survive unchanged into the next generation and keep the
            # score they were trained with; they are not retrained.
            for rank, parent in enumerate(parents):
                carried = dict(parent)
                carried.update({
                    "id": f"g{_pad(gen)}e{_pad(rank)}",
                    "idx": rank,
                    "gen": gen,
                    "parent": parent["id"],
                    "carried_over": True,
                })
                record_candidate(state, gen, carried)
                update_best(state, carried)

            work = []
            for rank, parent in enumerate(parents):
                for child in range(children_per_parent):
                    work.append({
                        "slot": elite_size + rank * children_per_parent + child,
                        "parent": parent,
                    })

        temperature = float(eureka.get("temperature_initial", 0.7) if gen == 0
                            else eureka.get("temperature_edit", 0.5))

        # ---- phase 1: obtain reward code for every slot of this generation ----
        # LLM calls stay sequential (they are latency bound and share one key);
        # training is what dominates wall-clock and is fanned out below.
        pending: List[Dict[str, Any]] = []
        for item in work:
            slot_index = item["slot"]
            parent = item["parent"]
            cid = f"g{_pad(gen)}c{_pad(slot_index)}"
            out_dir = candidate_dir(cfg, prefix, seed, gen, slot_index)
            for sub in ("llm_inputs", "prompt_records", "response_records", "validations"):
                (out_dir / sub).mkdir(parents=True, exist_ok=True)

            done = already_done(state, gen, slot_index)
            if done is not None:
                print(f"  [{cid}] already scored ({done['score']:.3f}) - skipping")
                continue

            print(f"\n  [{cid}] gen={gen} "
                  f"{'initial sample' if parent is None else 'child of ' + parent['id']}")

            parent_code = ""
            reflection = ""
            parent_score = 0.0
            if parent is not None:
                parent_code = read_text(parent["reward_path"])
                parent_train = Path(parent["train_dir"])
                reflection = build_reward_reflection(parent_train)
                parent_score = float(parent["score"])
                write_text(out_dir / "parent_reflection.md", reflection)

            reward_path = materialise_reward(
                cfg, client, ctx, out_dir,
                mode="initial" if parent is None else "edit",
                parent_code=parent_code, reflection=reflection,
                parent_score=parent_score, temperature=temperature,
            )
            pending.append({
                "id": cid, "idx": slot_index, "gen": gen,
                "parent": None if parent is None else parent["id"],
                "reward_path": str(reward_path),
                "train_dir": str(out_dir / "training"),
                "score": None,
            })

        # ---- phase 2: train this generation's candidates concurrently --------
        if pending:
            workers = max(1, min(parallel, len(pending)))
            print(f"\n  training {len(pending)} candidate(s) "
                  f"with {workers} concurrent run(s)", flush=True)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(train_candidate, config_path, cfg, prefix, seed,
                                item["gen"], item["idx"], Path(item["reward_path"]),
                                total_timesteps, eval_episodes): item
                    for item in pending
                }
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        train_dir = future.result()
                        item["train_dir"] = str(train_dir)
                        item["score"] = read_native_score(train_dir)
                    except Exception as exc:  # noqa: BLE001 - keep the population going
                        print(f"  [{item['id']}] training FAILED: {exc}")
                        item["score"] = None
                    record_candidate(state, gen, item)
                    update_best(state, item)
                    save_state(state_path, state)
                    print(f"  [{item['id']}] native score = {item['score']}", flush=True)

    # ---- summary -----------------------------------------------------------
    write_summary(root, cfg, state, prefix, seed, generations, total_timesteps,
                  eval_episodes, population_size, elite_size, children_per_parent)
    print(f"\nstate : {state_path}")
    print(f"summary: {root / 'eureka_summary.md'}")


def write_summary(root: Path, cfg, state: Dict[str, Any], prefix: str, seed: int,
                  generations: int, total_timesteps: int, eval_episodes: int,
                  population_size: int, elite_size: int, children: int) -> None:
    best = state.get("best")
    lines = [
        "# EUREKA-style population search - summary",
        "",
        f"- environment: `{cfg.get('training', {}).get('runner_env_id')}`",
        f"- generations: {generations}, population {population_size} "
        f"(elite {elite_size} x 1 + {children} children each)",
        f"- timesteps per candidate: {total_timesteps:,}",
        f"- evaluation episodes: {eval_episodes}",
        f"- selection metric: native `external_eval.mean_eval_reward`",
        f"- seed: {seed}",
        "",
    ]
    if best:
        lines += [f"**Best candidate: `{best['id']}` with native score {best['score']:.4f}**", ""]

    for block in sorted(state["generations"], key=lambda b: b["gen"]):
        lines += [f"## Generation {block['gen']}", "",
                  "| id | parent | native score |", "|---|---|---:|"]
        for cand in sorted(block["candidates"], key=lambda c: (c.get("score") is None,
                                                              -(c.get("score") or 0))):
            score = cand.get("score")
            lines.append(f"| {cand['id']} | {cand.get('parent') or '-'} | "
                         f"{'failed' if score is None else f'{score:.4f}'} |")
        lines.append("")

    if best:
        lines += ["## Best reward", "", f"`{best['reward_path']}`", ""]
        target = Path(root) / "best"
        target.mkdir(parents=True, exist_ok=True)
        write_text(target / "best_reward.py", read_text(best["reward_path"]))
        lines.append(f"copied to `{target / 'best_reward.py'}`")

    write_text(root / "eureka_summary.md", "\n".join(lines))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="EUREKA-style population reward search")
    ap.add_argument("--config", default="configs/env007_fragilecargo_eureka_baseline.yaml")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--population-size", type=int, default=None)
    ap.add_argument("--elite-size", type=int, default=None)
    ap.add_argument("--children-per-parent", type=int, default=None)
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=None)
    ap.add_argument("--parallel-candidates", type=int, default=None,
                    help="how many candidates of one generation to train at once "
                         "(they are independent; does not change results)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true",
                    help="run the whole flow without any LLM call")
    ap.add_argument("--resume", action="store_true",
                    help="reuse population.json and skip already scored candidates")
    ap.add_argument("--skip-context", action="store_true",
                    help="reuse an existing environment_card.md instead of regenerating it")
    args = ap.parse_args()

    run_eureka_population(
        config_path=args.config,
        prefix=args.prefix,
        generations=args.generations,
        population_size=args.population_size,
        elite_size=args.elite_size,
        children_per_parent=args.children_per_parent,
        total_timesteps=args.total_timesteps,
        eval_episodes=args.eval_episodes,
        parallel=args.parallel_candidates,
        seed=args.seed,
        mock=args.mock,
        resume=args.resume,
        skip_context=args.skip_context,
    )


if __name__ == "__main__":
    main()
