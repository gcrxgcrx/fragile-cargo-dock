# Subagent Research Signal

**Key Findings**: Final score=204.46, len=265.6, terminated=13/20, truncated=7/20. Reward range=[8.86,310.01]. Episode-sum reward dominated by terminal_success (195.0, 92.8% share) despite active_rate=0.2-0.4%.

**Component Anomalies**: terminal_success dominates at 92.8% signed/magnitude share. Dead: hard_hit and terminal_failure both 0% active. dock_enter rare (0.4% active) but 2.4% share. No high-magnitude self-cancelling component observed.

**Training Dynamics**: Across 11 checkpoints: terminal_success 0->145.7, total_reward 0.873->21.867, dock_enter 0.18->4.714, settle 0->3.055 (active 0%->32%), progress 1.246->4.075, terminal_failure -2.667->0. No plateau evident.

**Signal Quality**: Sparse terminal success gate drives nearly all reward; dense shaping components each <2.5% magnitude share. Summary abs_share all 0.0, so component share evidence comes from final-feedback table. hard_hit and terminal_failure gates never fire.

**Evidence Confidence**: `medium`
