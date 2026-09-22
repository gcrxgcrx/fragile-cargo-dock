# 设计理由

**第一步：信号覆盖审计**  
- 环境任务为“飞抵并平稳停靠目标平台”，终止条件包含 `body_not_awake_or_settled`（成功着陆）和 `crash_or_body_contact`/`horizontal_position_outside_viewport`（坠毁/飞出边界）。  
- 当前奖励完全缺少**迫近坠毁的预警信号**：没有对低空高下降速度的软惩罚，也没有利用支撑腿接触指示成功着陆。Agent 因无 crash 代价直接高速撞向平台， episode 长度仅 68.3 步即全部终止，得分极低。  
- 一句话结论：**缺 crash 前兆约束**。

**第二步：行为诊断**  
- Agent 在“快速失败”：从 `len=68.3`、`score=-121.77`、全部 `terminated` 推断，它大概率直接撞击平台而坠毁。  
- 当前 `progress` 鼓励缩短距离、降低合速度，但垂直高速下坠时，距离缩短带来的正向奖励可能超过速度增加带来的微弱惩罚，导致坠毁路径被奖励。  
- `fuel_cost` 对主发动机（`action==2`）的惩罚（-0.1）进一步抑制减速动作，使 agent 不愿点火制动。  
- **干预目标**：在低空对过高的下降速度施加软边界惩罚，让 agent 学会提前减速，避免直接撞地。

**第三步：选择干预层级**  
- 信号缺口明确，仅修改一个组件即可：添加“crash_prevention”惩罚。  
- 层级为 **Level 2 — 结构变换**（新增组件）。  
- 数学形式：  
  - `crash_risk = max(0, -ny_vel - 0.1)`（下降速度超过 0.1 的部分）  
  - `proximity = max(0, 1.0 - next_dist / 0.3)`（进入距目标 0.3 半径内时线性增大因子）  
  - `crash_prevention = -0.05 * crash_risk * proximity`  
- 系数校准：主信号 `progress` 的每步期望量级约 0.03（距离每步缩短 ~0.03，速度项更小），此惩罚在最坏情况（`proximity=1, crash_risk=0.5`）下约 -0.025，不到主信号的 1 倍；常态下更小，符合“总惩罚负担 ≤ 主信号 0.5x”要求。  
- 保留原有三个组件不变。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations (origin at target platform center/height)
    x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, l_contact, r_contact = obs
    nx_pos, ny_pos, nx_vel, ny_vel, n_angle, n_ang_vel, nl_contact, nr_contact = next_obs

    # Distance to target
    current_dist = (x_pos**2 + y_pos**2) ** 0.5
    next_dist    = (nx_pos**2 + ny_pos**2) ** 0.5

    # Speed magnitude
    current_speed = (x_vel**2 + y_vel**2) ** 0.5
    next_speed    = (nx_vel**2 + ny_vel**2) ** 0.5

    # ---------- Component A: potential-based shaping for approach + deceleration ----------
    w_dist = 1.0
    w_speed = 0.5
    phi_current = -(w_dist * current_dist + w_speed * current_speed)
    phi_next    = -(w_dist * next_dist    + w_speed * next_speed)
    progress = phi_next - phi_current

    # ---------- Component B: upright orientation penalty ----------
    orientation_penalty = -0.1 * (n_angle ** 2) - 0.05 * (n_ang_vel ** 2)

    # ---------- Component C: fuel cost ----------
    if action == 0:
        fuel_cost = 0.0
    elif action == 2:
        fuel_cost = -0.1
    else:   # 1 or 3
        fuel_cost = -0.05

    # ---------- Component D: crash prevention ----------
    # Penalize high downward speed when close to target platform.
    crash_risk = max(0.0, -ny_vel - 0.1)  # excess downward speed beyond 0.1
    proximity = max(0.0, 1.0 - next_dist / 0.3)  # linear ramp within 0.3 radius
    crash_prevention = -0.05 * crash_risk * proximity

    # ---------- Total ----------
    total = progress + orientation_penalty + fuel_cost + crash_prevention

    components = {
        "progress": progress,
        "orientation_penalty": orientation_penalty,
        "fuel_cost": fuel_cost,
        "crash_prevention": crash_prevention
    }
    return float(total), components
```

# 诊断摘要
- **audit**: 当前奖励漏了坠毁前兆信号（低空高下降速度）和着陆成功指示（支撑腿接触），导致 agent 快速撞击平台，没有平稳着陆引导。
- **behavior**: agent 在快速失败，68 步内全部终止，得分极负；它可能被 progress 奖励误导，直接高速坠向目标。
- **signal**: 缺少在近地处对下降速度的软约束（crash prevention），缺乏避免 crash 的梯度信息。
- **level**: Level 2 — 结构变换（新增组件 `crash_prevention`）
- **hypothesis**: 加入低空超速惩罚后，agent 将在接近目标时主动减速，避免直接撞地，从而延长 episode、获得更多 progress，并有机会学到双腿着陆，最终提高得分。
- **risk**: 如果惩罚过重，可能使 agent 畏缩不降、悬停耗尽步数而被截断；但当前系数轻微，且 progress 仍有向目标引导的驱动力，预计风险可控。若后续出现悬停倾向，可再调整 `fuel_cost` 或 `progress` 权重。