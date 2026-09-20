**evidence**：20/20 全部 truncated（len=400 满预算），无 terminated，说明策略既没成功也没触发失败终止；`crate_dock_alignment` 以 80.8% 的 magnitude_share 和 65% active_rate 主导奖励，但它只是"货箱朝向余弦"的持续状态值，与"把货箱推进坞"这一任务推进方向无关；真正的推进信号 `crate_to_dock_progress` 只有 2.8% 份额、episode_sum_mean=3.54（400 步内几乎没靠近），score=4.50 远低于 target=250。

**behavior_diagnosis**：策略学会了"原地把货箱朝向摆正"这一廉价 proxy——因为 `alignment_reward = 0.5 * align_factor * align_gate` 在 dist<0.3 时每步持续给分，而货箱朝向对齐不需要把它推进坞；同时 `crate_settling`（-12.16）和 `soft_contact_penalty`（-8.43）在惩罚"推动货箱所必需的速度与接触"，两者合计约 -20.6，压过了仅 +3.54 的推进奖励。净效果：最优策略是"靠近后停住、摆正朝向、不推"，即典型 proxy 徘徊。

**signal_completeness**：职责不完备。缺少"货箱真正进入坞内"的稠密可达信号——现有 progress 是逐步差分，在货箱被卡住或策略选择不推时恒为 0，无法提供持续梯度；而 alignment 被误当作主信号。obs[12]/obs[13] 已声明可用，可构造"货箱到坞距离"的稠密状态信号。

**selected_level**：Level 2（结构变换）。触发条件：主信号被 proxy 组件（alignment 持续状态值）支配，且辅助惩罚与任务推进方向对抗（违反校准规则 6）。

**selected_intervention**：重建主信号骨架——把 `crate_dock_alignment` 从"全局持续状态奖励"降级为**门控**（乘在推进信号上，仅在已接近坞时激活），并把主信号换成**货箱到坞距离的稠密改善量 + 距离凸化项**；同时把 `crate_settling` 与 `soft_contact_penalty` 的系数压到主信号 per-step 的 0.3x 以下，且 settling 只在 dist<0.1（真正入坞）时激活。

**falsifiable_hypothesis**：若 alignment 的 80.8% 份额确实是"不推也能刷分"的根源，则把它改为门控后，`crate_dock_alignment` 的 magnitude_share 应大幅下降，`crate_to_dock_progress` 的 episode_sum_mean 应显著上升，score 应超过 4.50。

**expected_next_round**：`crate_dock_alignment` magnitude_share 从 80.8% 降到 <20%；`crate_to_dock_progress` episode_sum_mean 从 3.54 升到 >15；`crate_settling` magnitude_share 从 9.7% 降到 <3%；score 从 4.50 升到 >10；len 仍为 400（truncated）或出现 terminated。

**main_risk**：把 alignment 改成门控后，若门控阈值设得太严，早期货箱远离坞时门控≈0，可能让策略失去"摆正朝向"的引导而更慢；另外若推进信号系数过大，可能诱导高速冲过坞（触发 crate_damaged 或滑出）。自检代入：①什么都不做、货箱静止在初始位置（dist≈0.5，crate_speed=0，contact=0）：progress=0，dist_term≈-0.5*0.5=-0.25，gate≈0，settling=0，contact=0 → 总≈-0.25；②正在把货箱推向坞（dist 从 0.5→0.48，crate_speed≈0.5，contact=1）：progress=+0.02*8=+0.16，dist_term≈-0.48*0.5=-0.24，gate 仍≈0，settling=0（dist>0.1），contact 惩罚=0（speed<1.0）→ 总≈-0.08。②(-0.08) > ①(-0.25)，满足要求。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (ndx * ndx + ndy * ndy) ** 0.5

    # 货箱速率（世界系，m/s）
    crate_speed = ((obs[8] * 3.0) ** 2 + (obs[9] * 3.0) ** 2) ** 0.5

    # 货箱朝向对齐度：cos(heading)，1=对齐
    align_cos = obs[10]
    align_factor = (align_cos + 1.0) * 0.5  # [0,1]

    contact = obs[14]

    # ---------- 主信号 A: 货箱到坞的稠密推进 ----------
    # 改善量（靠近为正）
    progress = (dist - next_dist)
    # 距离凸化项：越近奖励越大，提供持续梯度（避免卡住时梯度为 0）
    dist_term = -0.5 * dist
    # 对齐门控：仅在货箱已接近坞时启用，且随接近单调增强
    align_gate = 0.0
    if dist < 0.35:
        align_gate = max(0.0, 1.0 - dist / 0.35)
    # 门控因子：对齐越好，推进奖励越高（不阻断早期探索，最低 0.3）
    gate = 0.3 + 0.7 * align_factor * align_gate
    progress_reward = 8.0 * progress * gate + 0.5 * dist_term

    # ---------- 组件 B: 入坞对齐（仅在真正接近坞时，作为门控后的辅助） ----------
    # 只在 dist < 0.15 时给少量对齐奖励，且随接近增强
    align_reward = 0.0
    if dist < 0.15:
        align_reward = 0.3 * align_factor * (1.0 - dist / 0.15)

    # ---------- 组件 C: 入坞近静止（仅在 dist < 0.1 时，hinge） ----------
    settle_penalty = 0.0
    if dist < 0.1:
        settle_gate = 1.0 - dist / 0.1
        speed_excess = max(0.0, crate_speed - 0.05)
        settle_penalty = -0.5 * speed_excess * settle_gate

    # ---------- 组件 D: 边界安全（hinge，轻量） ----------
    cart_x = obs[0]
    cart_y = obs[1]
    boundary_penalty = 0.0
    if abs(cart_x) > 0.9:
        boundary_penalty -= 0.3 * (abs(cart_x) - 0.9)
    if abs(cart_y) > 0.9:
        boundary_penalty -= 0.3 * (abs(cart_y) - 0.9)
    sensor_max = max(obs[15], obs[16], obs[17])
    if sensor_max > 0.9:
        boundary_penalty -= 0.2 * (sensor_max - 0.9)

    # ---------- 组件 E: 接触冲击抑制（仅在接触且高速时，轻罚） ----------
    impact_penalty = 0.0
    if contact > 0.5:
        impact_excess = max(0.0, crate_speed - 1.5)
        impact_penalty = -0.2 * impact_excess

    # ---------- 汇总 ----------
    total_reward = (
        progress_reward
        + align_reward
        + settle_penalty
        + boundary_penalty
        + impact_penalty
    )

    components = {
        "crate_to_dock_progress": float(progress_reward),
        "crate_dock_alignment": float(align_reward),
        "crate_settling": float(settle_penalty),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components
```