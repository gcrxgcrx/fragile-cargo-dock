# ============================================================
# Ablation: Score-Only Feedback (No Component Evidence Table)
# ============================================================
# CONFIG          : configs/env001_ablation_score_only_v4.yaml
# SEEDS           : 5
# ROUNDS          : 10
# Memory          : ON
# Reflection      : structured (L1/L2/L3)
# Feedback        : SCORE-ONLY (no magnitude_share/active_rate)
#
# Purpose: Isolate the contribution of component-level evidence
# by removing magnitude_share, active_rate, and component
# breakdown from the reflection agent's feedback.
# Compare against ablation_eureka_feedback_v4 (main).
# ============================================================

Set-Location "d:\Code\python\research\form_github\expert-reward-agent"

# The API key is read from the environment and is never written to a file.
# Set it in your shell before running, e.g.:
#   $env:DEEPSEEK_API_KEY = "<your key>"
if (-not $env:DEEPSEEK_API_KEY) {
    throw "DEEPSEEK_API_KEY is not set in the environment."
}

$CONFIG = "configs/env001_ablation_score_only_v4.yaml"
$PREFIX = "ablation_score_only_v4"
$ROUNDS = 10
$TOTAL_TIMESTEPS = 1000000
$EVAL_EPISODES = 20

Write-Host "============================================================"
Write-Host " Ablation: Score-Only Feedback (No Component Evidence Table)"
Write-Host "============================================================"
Write-Host "CONFIG          : $CONFIG"
Write-Host "SEEDS           : 5"
Write-Host "ROUNDS          : $ROUNDS"
Write-Host "Memory          : ON"
Write-Host "Reflection      : structured (L1/L2/L3)"
Write-Host "Feedback        : SCORE-ONLY (no magnitude_share/active_rate)"
Write-Host ""

$seeds = @(0, 1, 2, 3, 4)
foreach ($seed in $seeds) {
    Write-Host ""
    Write-Host "##############################################################"
    Write-Host " SEED $seed / 5"
    Write-Host "##############################################################"

    python -m pipeline.run_iterative_experiment `
      --config $CONFIG `
      --prefix $PREFIX `
      --seed $seed `
      --rounds $ROUNDS `
      --total-timesteps $TOTAL_TIMESTEPS `
      --eval-episodes $EVAL_EPISODES
}

Write-Host ""
Write-Host "============================================================"
Write-Host " All seeds done."
Write-Host "============================================================"
