# 设计理由

## 信号覆盖审计结论
- **终止前兆缺失**：当前奖励对 crash（速度/姿态越界）没有有效前兆，对俯卧或走出视口也有惩罚盲区。
- **目标进度缺失**：任务要求“降落并稳定停靠”，但当前奖励只有距离缩短和绝对距离 shaping，完全没有接触、降落或稳定停靠的正面激励。
- **僵尸组件**：`velocity_constraint` 的 hinge 阈值过高（>1.5），在低速悬停时完全静默，变成一个无效惩罚；`orientation_penalty` 虽连续但力度偏弱。
- **一句话审计**：**缺失着陆成功信号，导致 agent 无限悬停，所有 episode 全部 truncation。**

## 行为诊断
- **agent 在做什么**：全部 episode 以长度为 1000 的 truncation 结束，得分低且范围集中。agent 学会朝目标中心下降，但到目标上方（y≈0、x≈0）后停止，既不触碰平台也不完成稳定停靠，依靠 `distance_shaping` 维持中等奖励，整体 net 为负。
- **干预目标**：**必须注入成功着陆的引导信号**，打破悬停均衡。
- **方向是否值得继续**：当前骨架（delta_distance + shaping）本身具备初期接近能力，只是缺少收尾的接触激励，有修补空间；前几轮其他骨架虽然加了接触奖励但可能因无安全门控而被利用，但我们规避 exploit 的方式是仅奖励“首次安全接触”——即事件奖励+严格条件。此方向尚可一试。

## 选择干预层级：**Level 2 — 结构变换**
**添加新组件 `safe_landing_bonus`**：一次性奖励，在 agent 从未接触到任一支撑腿触地时，若同时满足速度、倾角等安全条件，给予较大正向激励（+30）。这直接提供唯一缺失的“降落成功”信号，同时用严格安全门控避免为 crash 提供奖励。

### 数学形式
- 检测接触事件：`first_contact = (obs[6] == 0 and obs[7] == 0) and (next_obs[6] == 1 or next_obs[7] == 1)`
- 安全条件：
  - |x_vel| < 0.4
  - -0.4 < y_vel < 0.4
  - |body_angle| < 0.2
  - |angular_vel| < 0.2
- 奖励值：`safe = all(...)` → `bonus = 30.0 * first_contact * safe`

### 系数校准
- 主信号 per-step ≈ score/len ≈ -0.075，因此一次性 +30 是主信号绝对值的 400 倍，但这是稀疏事件，远低于 2× 平均每步主信号；不会支配轨迹。若成功触发，平均步均奖励会被大幅拉升，促使 agent 学习复现。
- 安全阈值设在软着陆的理想值附近（参考典型着陆条件），而非极端终止边界的 60‑80%，给 agent 足够的引导余量。
- 未触及任何已有惩罚/奖励的权重，保持其他组件不变。

# 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v1 + safe_landing_bonus: break the hovering equilibrium.
    Adds a one-time bonus for the first safe contact with the platform.
    """
    # --- Unpack observations ---
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_body_angle = next_obs[4]
    n_angular_vel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # --- Component A: delta_distance ---
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (nx_pos**2 + ny_pos**2) ** 0.5
    delta_distance = current_distance - next_distance
    progress_reward = 10.0 * delta_distance

    # --- Component B: bounded_distance_shaping ---
    distance_shaping = 2.0 * (1.0 / (1.0 + 0.5 * next_distance))

    # --- Component C: velocity_hinge_constraint ---
    h_speed = abs(nx_vel)
    h_penalty = max(0.0, h_speed - 2.0)
    v_speed = ny_vel
    v_down_penalty = max(0.0, -v_speed - 1.5)
    v_up_penalty = max(0.0, v_speed - 1.0)
    angular_penalty = max(0.0, abs(n_angular_vel) - 0.8)
    velocity_constraint = -0.5 * (h_penalty + v_down_penalty + v_up_penalty + angular_penalty)

    # --- Component D: orientation_stabilization ---
    orientation_penalty = -0.3 * (n_body_angle**2) - 0.2 * (n_angular_vel**2)

    # --- Component E: fuel_efficiency ---
    if action == 0:
        fuel_penalty = 0.0
    elif action == 2:
        fuel_penalty = -0.08
    else:
        fuel_penalty = -0.05

    # --- Component F: safe_landing_bonus (NEW) ---
    # One-time event bonus for first safe platform contact
    prev_contacted = 1.0 if (left_contact == 1.0 or right_contact == 1.0) else 0.0
    curr_contacted = 1.0 if (n_left_contact == 1.0 or n_right_contact == 1.0) else 0.0
    first_contact = 1.0 if (curr_contacted == 1.0 and prev_contacted == 0.0) else 0.0

    safe = (
        abs(nx_vel) < 0.4 and
        ny_vel > -0.4 and ny_vel < 0.4 and
        abs(n_body_angle) < 0.2 and
        abs(n_angular_vel) < 0.2
    )

    landing_bonus = 30.0 * first_contact * float(safe)

    # --- Assemble total reward ---
    total_reward = (
        progress_reward +
        distance_shaping +
        velocity_constraint +
        orientation_penalty +
        fuel_penalty +
        landing_bonus
    )

    components = {
        'progress_reward': progress_reward,
        'distance_shaping': distance_shaping,
        'velocity_constraint': velocity_constraint,
        'orientation_penalty': orientation_penalty,
        'fuel_penalty': fuel_penalty,
        'landing_bonus': landing_bonus
    }

    return float(total_reward), components
```

# 诊断摘要

- **audit**: 奖励函数缺少对“着陆并稳定停靠”目标的任何直接梯度，导致所有 episode 在目标区域上空悬停至 truncation。
- **behavior**: agent 在进度奖励驱动下靠近目标，但到达后悬停不动，既不接触平台也不结束回合，全程未触发任何成功终止，得分在浅负区间徘徊。
- **signal**: 缺安全着陆的成功信号；现有速度约束阈值太高，无法在着陆前引导减速；接触相关的引导完全空白。
- **level**: Level 2
- **hypothesis**: 添加一次性安全着陆奖励将提供唯一的正向终止激励，引导 agent 改变末端行为，从“悬停”转向“满足安全条件触碰平台”，最终触发稳定停靠终止，使分数由负转正。
- **risk**: 如果 agent 在探索中始终无法同时满足接触与严格安全阈值（例如因随机动作导致触地必 crash），则 `landing_bonus` 可能长期为 0，训练仍会悬停。但配合已有的姿态和速度惩罚，agent 有动力在接近目标时减速，故安全条件可达。若出现停滞，后续可降低安全阈值或改为渐进式接触奖励。