# Response Record

1. `evidence`：score=-6.74，19/20 truncated（超时存活），无 early_terminal；`dock_approach_improvement` 与 `fragile_impact_penalty` 的 active_rate=0.0%，`joint_dock_completion` 占 magnitude 98.6% 且 episode_sum_mean=87.7——主信号完全失效，唯一活跃组件是一个与任务进展无关的静态状态量。
2. `behavior_diagnosis`：策略发现"停在 dock 附近不动"即可持续领取 joint 奖励（f_near 高、f_slow 高、f_align 高），于是放弃推箱；货箱从未真正移动，`dock_approach_improvement` 恒为 0 证明货箱到 dock 距离从未变化。
3. `signal_completeness`：职责严重缺失——没有"货箱实际位移"信号，没有"货箱进入 dock 矩形"的联合完成信号，没有"低速停稳"的收敛信号；现有 joint 是纯状态值，占据即得分，属典型 state→improvement 塌缩。
4. `selected_level`：Level 3 重建（累积记录同骨架连续 3 轮未刷新 best=3.86，且历史最佳 3.86 < target×0.5=125）。
5. `selected_intervention`：更换主信号框架——以 `improvement_delta`（货箱到 dock 距离减少量）为主信号，叠加 `joint_condition_proxy`（near×slow×align 几何平均，仅作辅助且权重压低），并新增基于 obs[15..17] 的局部障碍 hinge 惩罚替代全局边界惩罚。
6. `falsifiable_hypothesis`：若主信号改为"货箱向 dock 的实际位移"，则策略必须推动货箱才能获得正奖励，`dock_approach_improvement` 的 active_rate 应从 0% 升至 >50%，且 episode_sum_mean 显著为正；若下一轮该组件仍为 0，则说明货箱位移信号不可达或权重过低。
7. `expected_next_round`：`dock_approach_improvement` active_rate > 50%、episode_sum_mean > 0；`joint_dock_completion` magnitude_share 从 98.6% 降至 <40%；score 从 -6.74 提升至 > 0；len 保持 ~400（不引入快速失败）。
8. `main_risk`：improvement_delta 在 dock 附近会震荡（推过头再推回），可能产生正负抵消；且若货箱初始就在 dock 附近，delta 会迅速饱和——需靠 joint 辅助项提供停稳梯度。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- unpack ----------
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    cvx = obs[8]
    cvy = obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    crate_cos = obs[10]
    crate_sin = obs[11]

    cart_fwd = obs[4]
    contact = obs[14]
    cart_x = obs[0]
    cart_y = obs[1]

    s_front = obs[15]
    s_left = obs[16]
    s_right = obs[17]

    # ---------- A. MAIN: crate_to_dock_progress (improvement_delta) ----------
    # 货箱到 dock 的距离（归一化坐标）
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new  # >0 表示货箱确实向 dock 靠近

    # 主信号：对称，靠近给正分，远离给负分
    w_progress = 30.0
    r_progress = w_progress * progress

    # ---------- B. AUX: joint_dock_completion (joint_condition_proxy, 几何平均) ----------
    # 三因子：near / slow / align，每因子连续 bounded，带 floor 防塌缩
    f_near = max(0.0, 1.0 - dist_new / 0.5)
    f_slow = 1.0 / (1.0 + 40.0 * crate_speed)
    f_align = abs(crate_cos)

    f_near_c = max(f_near, 0.05)
    f_slow_c = max(f_slow, 0.05)
    f_align_c = max(f_align, 0.05)
    joint = (f_near_c * f_slow_c * f_align_c) ** (1.0 / 3.0)

    # 辅助权重压低，避免占据状态即刷分
    w_joint = 0.4
    r_joint = w_joint * joint

    # ---------- C. fragile_impact_penalty (hinge, 仅接触时) ----------
    rel_speed = abs(cart_fwd - crate_speed)
    impact_excess = max(0.0, rel_speed - 0.15)
    w_impact = 4.0
    r_impact = -w_impact * impact_excess * contact

    # ---------- D. local_obstacle_penalty (hinge, 局部障碍接近度) ----------
    # 只在接近度超过 0.7 时惩罚，避免全时惩罚压制探索
    obs_excess = max(0.0, s_front - 0.7) + max(0.0, s_left - 0.7) + max(0.0, s_right - 0.7)
    w_obs = 3.0
    r_obs = -w_obs * obs_excess

    # ---------- total ----------
    total_reward = r_progress + r_joint + r_impact + r_obs

    components = {
        "crate_to_dock_progress": r_progress,
        "joint_dock_completion": r_joint,
        "fragile_impact_penalty": r_impact,
        "local_obstacle_penalty": r_obs,
    }

    return float(total_reward), components
```
