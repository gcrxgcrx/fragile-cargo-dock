# Response Record

1. `evidence`：20/20 全 truncated（len=400），无 terminated，说明策略从未触发成功或失败终止；`dock_completion_joint` episode_sum_mean=4396 且 magnitude_share=100%，但 score=-1.94，说明该 proxy 被"占据近 dock 状态"刷满却与外部成功完全脱钩；其余三个组件 active_rate=0，是僵尸组件；历史 best=3.86 已连续 6 轮未刷新。

2. `behavior_diagnosis`：策略学会了"把货箱推到 dock 附近并保持低速/对齐的静态姿态"来持续领取 joint 奖励，但从未完成"完整进入 dock + 朝向 <30° + 速度 <0.05 + 连续 10 步"的联合条件，属于典型 proxy 徘徊（state-occupation exploit）。

3. `signal_completeness`：主信号职责（联合完成）存在但形态错误——用"状态值"而非"改善量/完成度"，导致占据即得分；缺少对"货箱是否真正进入 dock 矩形"的几何门控；速度/朝向因子被 floor 抬到 0.05 后失去区分度。

4. `selected_level`：Level 3 重建（触发条件：同骨架族 ≥4 轮、连续 ≥3 轮未刷新 best、Level 2 形态变换后得分无实质改善）。

5. `selected_intervention`：更换主信号框架——用 `improvement_delta`（货箱到 dock 距离的逐步减少）作为唯一主学习信号，叠加 `joint_condition_proxy`（几何平均，无 floor 塌缩）作为**完成度门控**乘子，并新增基于 obs[15..17] 的局部障碍 hinge 惩罚；删除所有 state-occupation 型奖励。

6. `falsifiable_hypothesis`：若失败原因是"占据近 dock 状态即得分"，则改为 improvement_delta 主信号 + 完成度门控后，`dock_progress_delta` 的 episode_sum_mean 应显著非零且与 score 同向，`joint_completion_gate` 的 magnitude_share 应下降，score 应上升。

7. `expected_next_round`：`dock_progress_delta` active_rate > 50% 且 episode_sum_mean > 0；`joint_completion_gate` magnitude_share < 60%；score > 3.86（刷新 best）；len 仍约 400（truncated 为主，符合任务难度）。

8. `main_risk`：improvement_delta 在目标附近震荡会退化为噪声，且门控乘子可能再次塌缩为 0 导致梯度消失；用几何平均 + 0.1 floor 缓解，但若 score 仍不升则需回到纯 delta 主信号。

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

    # ---------- A. MAIN: dock_progress_delta (improvement_delta) ----------
    # 货箱到 dock 距离的逐步减少量，唯一主学习信号
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new
    w_progress = 40.0
    r_progress = w_progress * progress

    # ---------- B. GATE: joint_completion_gate (几何平均, 无 floor 塌缩) ----------
    # 三因子：near / slow / align，连续 bounded，几何平均防塌缩
    f_near = max(0.0, 1.0 - dist_new / 0.6)
    f_slow = 1.0 / (1.0 + 30.0 * crate_speed)
    f_align = abs(crate_cos)

    # 几何平均：任一因子为 0 时整体为 0，但用 0.1 下限避免完全塌缩
    f_near_c = max(f_near, 0.1)
    f_slow_c = max(f_slow, 0.1)
    f_align_c = max(f_align, 0.1)
    gate = (f_near_c * f_slow_c * f_align_c) ** (1.0 / 3.0)

    # 门控乘子：完成度越高，主信号越被放大（不独立给分）
    w_gate = 15.0
    r_gate = w_gate * gate * max(0.0, progress)

    # ---------- C. crate_speed_hinge (停稳引导, 只在货箱过快时罚) ----------
    speed_excess = max(0.0, crate_speed - 0.15)
    w_speed = 4.0
    r_speed = -w_speed * speed_excess

    # ---------- D. fragile_impact_penalty (hinge, 仅接触时) ----------
    rel_speed = abs(cart_fwd - crate_speed)
    impact_excess = max(0.0, rel_speed - 0.25)
    w_impact = 2.0
    r_impact = -w_impact * impact_excess * contact

    # ---------- E. local_obstacle_penalty (hinge, 基于 obs[15..17]) ----------
    # 只在接近静态障碍时罚，给探索留空间
    obs_front = obs[15]
    obs_left = obs[16]
    obs_right = obs[17]
    obs_max = max(obs_front, obs_left, obs_right)
    obstacle_excess = max(0.0, obs_max - 0.7)
    w_obstacle = 2.0
    r_obstacle = -w_obstacle * obstacle_excess

    # ---------- total ----------
    total_reward = r_progress + r_gate + r_speed + r_impact + r_obstacle

    components = {
        "dock_progress_delta": r_progress,
        "joint_completion_gate": r_gate,
        "crate_speed_hinge": r_speed,
        "fragile_impact_penalty": r_impact,
        "local_obstacle_penalty": r_obstacle,
    }

    return float(total_reward), components
```
