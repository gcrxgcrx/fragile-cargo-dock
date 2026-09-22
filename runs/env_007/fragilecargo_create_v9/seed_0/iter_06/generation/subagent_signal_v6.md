# Subagent Research Signal

**Key Findings**: mean_eval_reward=309.5, terminated=20/20, len=180.9; terminal_success=95.5% share (300/314) at 0.6% active.

**Component Anomalies**: terminal_success dominates >70%; hard_hit and terminal_failure dead (0% active); dock_enter rare (0.6% active) but mean 5.0.

**Training Dynamics**: 11 ckpts: terminal_success 0->264, dock_enter 0.18->4.5, progress 1.38->3.84, settle 0.004->2.38, total 1.34->28.75.

**Signal Quality**: score gated by rare terminal_success/dock_enter; dense approach/progress/time_cost active but tiny magnitude; hard_hit/terminal_failure thresholds never crossed.

**Evidence Confidence**: `high`
