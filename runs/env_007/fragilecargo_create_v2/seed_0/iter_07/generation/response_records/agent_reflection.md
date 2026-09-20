# Response Record

**evidence**：6 轮迭代全部失败，best=17.69（iter4），当前 11.18；18/20 truncated、仅 2/20 terminated，len≈388 说明 agent 长期存活但从未完成交付；`crate_to_dock_progress` episode_sum_mean=8.48 但 signed_share 72.9%，说明它只是"靠近-远离"的净小量，不是稳定推进；`completion_improvement` active_rate 仅 32.2%，`boundary_avoidance` active_rate 0.2%（僵尸组件）。

**behavior_diagnosis**：agent 学会了"在货箱附近低速徘徊"——它把货箱推近坞一点点就停手，因为主信号是**距离改善量**，一旦停下不再靠近，改善量归零，但也没有任何信号要求它"保持货箱在坞内静止 10 步"。它从未触发 `docked_success`（2/20 terminated 大概率是越界/损坏而非成功）。

**signal_completeness**：职责不全。缺"联合完成度"的**状态值**引导——现有信号全是 delta，agent 无法感知"我现在离成功有多近"。obs[10]/[11]（货箱朝向）、obs[8]/[9]（货箱速度）、obs[12]/[13]（到坞偏移）三个成功条件维度都可用但从未被组合成"当前完成度"。

**selected_level**：Level 3 重建（累积记录连续 6 轮未刷新 best，同骨架族迭代 ≥4 轮且 best 17.69 < 250×0.5）。

**selected_intervention**：更换主信号框架为 **joint_condition_proxy（几何平均）**：把三个成功条件（坞内位置、朝向对齐、近静止）各自做成连续 bounded factor，用几何平均合成"完成度状态值"，再对**完成度状态值**做凸化奖励（而非 delta）。同时保留一个小的距离改善 delta 作为早期引导。

**falsifiable_hypothesis**：如果失败原因是"agent 不知道当前离成功多近、缺乏联合满足引导"，那么用几何平均完成度状态值作为主信号后，agent 会主动把货箱推向"坞内+对齐+静止"的联合区域，`docked_success` 终止数应从 2/20 上升，score 应超过 17.69。

**expected_next_round**：`joint_completion` 组件 active_rate > 80%，episode_sum_mean 显著 > 0；terminated 数 ≥ 4/20；score > 17.69（刷新 best）。

**main_risk**：几何平均中"近静止"因子可能诱导 agent 停在远处不动（静止但不在坞内）。**自检**：①什么都不做、货箱静止在初始位置（dist≈1.0）：dock_factor≈0，几何平均≈0，总奖励≈0；②正在把货箱推向坞（dist 从 0.5→0.4，货箱有速度）：dock_factor≈0.33，align≈0.5，slow≈0.5，几何平均≈0.44，凸化后 ≈0.44²×20≈3.9，加上 delta 引导 >0。②严格高于①，通过。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (ndx * ndx + ndy * ndy) ** 0.5

    # 货箱世界系速率 (m/s)
    cvx = obs[8] * 3.0
    cvy = obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向对齐度：cos(heading)，1=对齐
    align_cos = obs[10]
    align_factor = (align_cos + 1.0) * 0.5  # [0,1]

    contact = obs[14]

    # ---------- 主信号 A: 联合完成度状态值（几何平均，非塌缩） ----------
    # 三个成功条件各自连续 bounded factor
    # 1) 坞内位置：dist 越小越接近 1（终止边界约 0.15，用 0.6 给足梯度）
    dock_factor = max(0.0, 1.0 - dist / 0.6)
    # 2) 朝向对齐：align_factor 已在 [0,1]
    # 3) 近静止：速度越小越接近 1
    slow_factor = 1.0 / (1.0 + 3.0 * crate_speed)

    # 几何平均：任一因子为 0 时整体为 0，但比裸乘积平滑
    joint_completion = (dock_factor * align_factor * slow_factor) ** (1.0 / 3.0)

    # 凸化：强化高完成度区域的梯度，打破低水平稳态
    completion_reward = 20.0 * (joint_completion ** 2)

    # ---------- 主信号 B: 距离改善 delta（早期引导，bounded 防极端） ----------
    raw_progress = dist - next_dist  # 靠近为正
    progress = raw_progress / (1.0 + abs(raw_progress) * 20.0)
    progress_reward = 6.0 * progress

    # ---------- 组件 C: 边界安全（hinge，阈值 0.85） ----------
    boundary_penalty = 0.0
    if abs(obs[0]) > 0.85:
        boundary_penalty -= 0.3 * (abs(obs[0]) - 0.85)
    if abs(obs[1]) > 0.85:
        boundary_penalty -= 0.3 * (abs(obs[1]) - 0.85)
    sensor_max = max(obs[15], obs[16], obs[17])
    if sensor_max > 0.85:
        boundary_penalty -= 0.2 * (sensor_max - 0.85)

    # ---------- 组件 D: 接触冲击抑制（仅在接触且高速时，轻罚） ----------
    impact_penalty = 0.0
    if contact > 0.5:
        impact_excess = max(0.0, crate_speed - 1.5)
        impact_penalty = -0.2 * impact_excess

    # ---------- 汇总 ----------
    total_reward = (
        completion_reward
        + progress_reward
        + boundary_penalty
        + impact_penalty
    )

    components = {
        "joint_completion": float(completion_reward),
        "crate_to_dock_progress": float(progress_reward),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components
```
