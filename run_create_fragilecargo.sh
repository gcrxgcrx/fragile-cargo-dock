#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# CREATE on FragileCargoDock-v0 (env_007).
#
#   CREATE      : this script  -> pipeline.run_iterative_experiment
#                 single lineage, structured diagnosis, semantic-localised
#                 edit, intervention memory, best-reward archive
#   EUREKA base : run_eureka_fragilecargo.sh -> pipeline.run_eureka_population
#                 population of sampled rewards, reflection-driven edit,
#                 selection by native score
#
# Both run on the same environment and report the same selection metric
# (external_eval.mean_eval_reward). Keep the LLM settings identical between the
# two, otherwise the comparison is not controlled:
#   - model      : deepseek-flash (DeepSeek-V4.1-Flash) in both configs
#   - thinking   : DEEPSEEK_THINKING=disabled in both (the EUREKA baseline run
#                  in runs/env_007/fragilecargo_eureka used disabled)
#   - budget     : 10 runs x total_timesteps in both (30M environment steps)
#
# Prerequisite
# ------------
#   export DEEPSEEK_API_KEY="<your key>"
#
# Run from the repository root.
# ---------------------------------------------------------------------------
set -euo pipefail

CONFIG="configs/env007_fragilecargo_eureka.yaml"
PREFIX="${PREFIX:-fragilecargo_create}"
ROUNDS="${ROUNDS:-10}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-3000000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SEEDS="${SEEDS:-0}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "ERROR: DEEPSEEK_API_KEY is not set." >&2
  echo "       export DEEPSEEK_API_KEY=\"<your key>\" before running." >&2
  exit 1
fi

# Match the EUREKA baseline. Thinking mode also risks returning an empty
# `content` with finish_reason='length' when the chain of thought exhausts
# max_tokens, which aborts a long unattended run.
export DEEPSEEK_THINKING="${DEEPSEEK_THINKING:-disabled}"

echo "============================================================"
echo " CREATE - single-lineage structured reward search"
echo "============================================================"
echo "CONFIG          : $CONFIG"
echo "PREFIX          : $PREFIX"
echo "SEEDS           : $SEEDS"
echo "ROUNDS          : $ROUNDS"
echo "TIMESTEPS/ROUND : $TOTAL_TIMESTEPS"
echo "EVAL EPISODES   : $EVAL_EPISODES"
echo "Selection       : native external_eval.mean_eval_reward"
echo "Model           : deepseek-flash (DeepSeek-V4.1-Flash)"
echo "Thinking mode   : $DEEPSEEK_THINKING"
echo "Target score    : see iteration.target_score in the config"
echo ""

for seed in $SEEDS; do
  echo ""
  echo "##############################################################"
  echo " SEED $seed"
  echo "##############################################################"
  python -m pipeline.run_iterative_experiment \
    --config "$CONFIG" \
    --prefix "$PREFIX" \
    --seed "$seed" \
    --rounds "$ROUNDS" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES"
done

echo ""
echo "============================================================"
echo " Done. Results under runs/env_007/$PREFIX/seed_<seed>/"
echo "   experiment_summary.md   per-round scores + stop reason"
echo "   memory/reward_memory.md intervention memory"
echo "   best/best_reward.py     best reward found"
echo "   iter_XX/                per-round generation + training"
echo "============================================================"
