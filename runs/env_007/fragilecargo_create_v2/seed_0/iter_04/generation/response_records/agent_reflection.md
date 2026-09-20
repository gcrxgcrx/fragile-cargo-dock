# Response Record

**evidence**：iter3 全 20 回合 truncated（len=400），无 terminated，说明 agent 既没成功也没失败，只是存活；`joint_completion_proxy` episode_sum_mean=696（magnitude_share 98.9%，active_rate 73.2%）却只换来 score=2.13，是典型 proxy 刷分——它在 `dist<0.5` 时用 `f_slow`（货箱近静止）持续收分，agent 学会"把货箱推到坞附近然后停住不动"，而非完成交付；`crate_to_dock_progress` 仅 5.48（0.9%），说明真正推进几乎没发生。

**behavior_diagnosis**：策略退化为"把货箱挪到坞区边缘后保持静止"的徘徊刷分行为，主信号被一个与任务完成不同向的静止因子支配，任务推进（货箱真正进入坞并稳定交付）从未发生。

**signal_completeness**：职责不完备——缺少"货箱真正进入坞内（完全进入）"的几何判据，也缺少"朝向对齐"与"近静止"作为**门控**而非独立收分项；现有 proxy 把静止当成奖励本身，与"推进任务必然产生速度"对抗。

**selected_level**：Level 3 重建（累积记录同骨架连续 3 轮未刷新 best=4.50，且历史最佳远低于 target×0.5=125）。

**selected_intervention**：更换主信号框架——用 `improvement_delta`（货箱到坞距离的逐步减少）作为唯一主推进信号，并把"坞内 + 对齐 + 近静止"做成**门控乘子**（`soft_health_gate` 形式）乘在推进信号上，而非独立持续奖励；删除会奖励静止的 `joint_completion_proxy`。

**falsifiable_hypothesis**：若失败源于"静止刷分"，则移除静止收分项、改为距离改善量后，`crate_to_dock_progress` 的 magnitude_share 应显著上升，且 score 应超过 4.50（历史 best）。

**expected_next_round**：`crate_to_dock_progress` magnitude_share 从 0.9% 升至 >50%；`joint_completion_proxy` 消失或占比 <20%；score > 4.50；len 仍接近 400（未引入快速失败）。

**main_risk**：距离改善量在坞附近会震荡（来回推），可能产生小幅正负抵消；若门控过严，早期探索被压制导致推进信号稀疏。

**自检代入**：
- ① 什么都不做、货箱静止在初始位置：`progress=0`，`gate` 因 dist 大而≈0 → 单步总奖励 ≈ 0。
- ② 正在把货箱推向坞（货箱有速度、dist 减小）：`progress>0`，`gate` 随接近而增大 → 单步总奖励 > 0。
- ② > ①，满足要求。

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

    # ---------- 主信号：货箱到坞距离的改善量（唯一推进信号） ----------
    progress = dist - next_dist  # 靠近为正
    progress_reward = 10.0 * progress

    # ---------- 门控：接近坞 + 对齐 + 近静止，作为乘子而非独立收分 ----------
    # 接近门：dist 越小越接近 1（阈值 0.6，终止边界约 0.15 的 4x 缓冲）
    near_gate = max(0.0, 1.0 - dist / 0.6)
    # 对齐门：朝向越对齐越接近 1（下限 0.2 防塌缩）
    align_gate = max(0.2, align_factor)
    # 静止门：货箱越慢越接近 1（下限 0.2 防塌缩）
    slow_gate = max(0.2, 1.0 - crate_speed / 0.6)
    # 几何平均，避免乘积塌缩
    gate = (near_gate * align_gate * slow_gate) ** (1.0 / 3.0)

    # 门控只放大"正在推进"的信号：推进为正时按门控加成
    if progress > 0.0:
        gated_progress = progress_reward * (1.0 + 2.0 * gate)
    else:
        gated_progress = progress_reward

    # ---------- 组件 C: 边界安全（hinge，阈值 0.85 = 终止边界 1.0 的 85%） ----------
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
        gated_progress
        + boundary_penalty
        + impact_penalty
    )

    components = {
        "crate_to_dock_progress": float(gated_progress),
        "dock_gate": float(gate),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components
```
