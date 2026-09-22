# Subagent Research Signal

**Key Findings**: mean_eval_reward=309.55, terminated=20/20, len=180.9; terminal_success episode_sum_mean=300.0=95.5% signed share at active_rate=0.6%.

**Component Anomalies**: terminal_success dominates 95.5%; hard_hit/terminal_failure active 0%; dock_enter active 0.6%; action_cost and time_cost 100% active but signed shares -0.0%/-0.1%.

**Training Dynamics**: 11 checkpoints: terminal_success 0->264.0, total_reward 1.342->28.752, dock_enter 0.180->4.500, settle 0.004->2.376 (active 0%->16%), progress 1.376->3.840.

**Signal Quality**: dead gates: hard_hit/terminal_failure 0% active; early_terminal=0/20; terminal_success sparse (0.6%) but huge; score_range [309.12,310.57] narrow.

**Evidence Confidence**: `high`
