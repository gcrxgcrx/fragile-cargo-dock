#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# EUREKA-style population reward search on FragileCargoDock-v0.
#
# This is the CREATE *baseline* (M3: population reflection search), not CREATE.
# It samples several reward candidates per generation, trains each of them, and
# selects by the environment's native score. The only feedback given back to
# the LLM is the reward reflection (task score + how the reward's own component
# values evolved during training) - no structured diagnosis, no memory, no
# diagnostic subagent.
#
#   CREATE      : python -m pipeline.run_iterative_experiment   (single lineage)
#   EUREKA base : python -m pipeline.run_eureka_population      (this script)
#
# Prerequisite
# ------------
# The LLM key for this flow is read from EUREKA_DEEPSEEK_API_KEY. It is never
# written to any file in this repository, so export it in your shell first:
#
#   export EUREKA_DEEPSEEK_API_KEY="<your key>"
#
# Run from the repository root.
# ---------------------------------------------------------------------------
set -euo pipefail

CONFIG="configs/env007_fragilecargo_eureka_baseline.yaml"
PREFIX="fragilecargo_eureka"
GENERATIONS="${GENERATIONS:-4}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEEDS="${SEEDS:-0}"

if [[ -z "${EUREKA_DEEPSEEK_API_KEY:-}" ]]; then
  echo "ERROR: EUREKA_DEEPSEEK_API_KEY is not set." >&2
  echo "       export EUREKA_DEEPSEEK_API_KEY=\"<your key>\" before running." >&2
  exit 1
fi

# Thinking mode ignores `temperature`, and this flow needs temperature for the
# generation-0 population diversity. It also risks returning an empty `content`
# with finish_reason='length' when the chain of thought exhausts max_tokens.
export DEEPSEEK_THINKING=disabled

# Population shape must satisfy population_size = elite_size * (1 + children).
POPULATION_SIZE="${POPULATION_SIZE:-4}"
ELITE_SIZE="${ELITE_SIZE:-2}"
CHILDREN_PER_PARENT="${CHILDREN_PER_PARENT:-1}"

echo "============================================================"
echo " EUREKA-style Population Search (CREATE baseline M3)"
echo "============================================================"
echo "CONFIG          : $CONFIG"
echo "PREFIX          : $PREFIX"
echo "SEEDS           : $SEEDS"
echo "GENERATIONS     : $GENERATIONS"
echo "POPULATION      : $POPULATION_SIZE (elite $ELITE_SIZE x (1 + $CHILDREN_PER_PARENT))"
echo "TIMESTEPS/CAND  : $TOTAL_TIMESTEPS"
echo "EVAL EPISODES   : $EVAL_EPISODES"
echo "Selection       : native external_eval.mean_eval_reward"
echo "Feedback        : reward reflection only (score + component values)"
echo "LLM key         : EUREKA_DEEPSEEK_API_KEY (from environment)"
echo ""

for seed in $SEEDS; do
  echo ""
  echo "##############################################################"
  echo " SEED $seed"
  echo "##############################################################"
  python -m pipeline.run_eureka_population \
    --config "$CONFIG" \
    --prefix "$PREFIX" \
    --seed "$seed" \
    --generations "$GENERATIONS" \
    --population-size "$POPULATION_SIZE" \
    --elite-size "$ELITE_SIZE" \
    --children-per-parent "$CHILDREN_PER_PARENT" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES"
done

echo ""
echo "============================================================"
echo " Done. Results under runs/env_007/$PREFIX/seed_<seed>/"
echo "   population.json      full search state"
echo "   eureka_summary.md    per-generation scores + best candidate"
echo "   best/best_reward.py  best reward found"
echo "============================================================"
