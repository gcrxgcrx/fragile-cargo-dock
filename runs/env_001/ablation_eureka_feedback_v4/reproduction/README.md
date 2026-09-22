# Reproduce `ablation_eureka_feedback_v4` Seed 4

This folder contains reproduction instructions for the experiment in:

- `runs/env_001/ablation_eureka_feedback_v4`

## Recommended reproduction commands

### 1) Reproduce the full iterative experiment for seed 4

From the repo root (`d:\Code\python\research\form_github\expert-reward-agent`):

```powershell
python -m pipeline.run_iterative_experiment \
  --config configs/env001_ablation_eureka_feedback_v4.yaml \
  --prefix ablation_eureka_feedback_v4 \
  --seed 4 \
  --rounds 10 \
  --total-timesteps 1000000 \
  --eval-episodes 20
```

This reruns the same configuration used by the ablation study and should reproduce the iterative reward generation + training process for seed 4.

### 2) Reproduce only the final best reward training for seed 4

If you only want to retrain/evaluate the best discovered reward function from iteration 8:

```powershell
python -m training.train_sb3_wrapper \
  --config configs/env001_ablation_eureka_feedback_v4.yaml \
  --reward runs/env_001/ablation_eureka_feedback_v4/seed_4/best/best_reward.py \
  --run-name ablation_eureka_feedback_v4/seed_4/iter_08/training \
  --save-dir runs/env_001/ablation_eureka_feedback_v4/seed_4/iter_08/training \
  --total-timesteps 1000000 \
  --eval-episodes 20 \
  --seed 4
```

This uses the same training wrapper as the experiment and will produce the same `training_summary.json` and `train_config_used.yaml` format.

## Key files referenced

- config: `configs/env001_ablation_eureka_feedback_v4.yaml`
- best reward: `runs/env_001/ablation_eureka_feedback_v4/seed_4/best/best_reward.py`
- seed 4 best summary: `runs/env_001/ablation_eureka_feedback_v4/seed_4/best/best_summary.md`
- exact training record: `runs/env_001/ablation_eureka_feedback_v4/seed_4/iter_08/training/train_config_used.yaml`

## Notes

- The full experiment run uses the same iterative pipeline and LLM configuration as the original ablation.
- If the pipeline requires `DEEPSEEK_API_KEY`, set it in your environment before running the full experiment.
- The `run_iterative_experiment` command is the same driver used by repo scripts such as `run_ablation_eureka_feedback_v4.sh`.
