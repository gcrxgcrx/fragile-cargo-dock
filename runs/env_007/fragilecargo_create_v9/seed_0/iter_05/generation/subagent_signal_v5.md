# Subagent Research Signal

**Key Findings**: score=263.78, terminated=17/20, len=201.95; terminal_success signed_share=95.2% (mean 255, active 0.4%) dominates; total_reward 0.1018 vs original_env_reward 0.9250.

**Component Anomalies**: terminal_success >70% share (95.2%). Dead: hard_hit and terminal_failure both mean=0.0, nonzero=0.0%. dock_enter nonzero=0.3% yet dynamics grow 0.240->4.722.

**Training Dynamics**: terminal_success 0.000->266.667; total_reward 1.581->29.694 across 11 ckpts; action_cost decays -0.195->-0.096; terminal_failure -0.533->0.000; progress 1.469->3.989, active 68%->75%.

**Signal Quality**: All training-summary abs_share=0.0% (reporting artifact). score_range=[1.26,310.20] wide; 3/20 truncated; dock_enter/settle active-rate mismatch between summary (0.3%/14.7%) and feedback (0.4%/18.3%).

**Evidence Confidence**: `medium`
