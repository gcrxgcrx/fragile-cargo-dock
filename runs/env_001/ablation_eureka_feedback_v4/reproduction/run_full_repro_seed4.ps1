Set-Location "d:\Code\python\research\form_github\expert-reward-agent"

# The API key is read from the environment and is never written to a file.
# Set it in your shell before running, e.g.:
#   $env:DEEPSEEK_API_KEY = "<your key>"
if (-not $env:DEEPSEEK_API_KEY) {
    throw "DEEPSEEK_API_KEY is not set in the environment."
}

python -m pipeline.run_iterative_experiment `
  --config configs/env001_ablation_eureka_feedback_v4.yaml `
  --prefix ablation_eureka_feedback_v4 `
  --seed 4 `
  --rounds 10 `
  --total-timesteps 1000000 `
  --eval-episodes 20
