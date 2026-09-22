Set-Location "d:\Code\python\research\form_github\expert-reward-agent"

python -m training.train_sb3_wrapper `
  --config configs/env001_ablation_eureka_feedback_v4.yaml `
  --reward runs/env_001/ablation_eureka_feedback_v4/seed_4/best/best_reward.py `
  --run-name ablation_eureka_feedback_v4/seed_4/iter_08/training `
  --save-dir runs/env_001/ablation_eureka_feedback_v4/seed_4/iter_08/training `
  --total-timesteps 1000000 `
  --eval-episodes 20 `
  --seed 4
