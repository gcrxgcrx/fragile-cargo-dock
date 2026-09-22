# 设计理由

## 信号覆盖审计
- **终止→前兆**: 环境声明了 `body_not_awake_or_settled`（潜在成功）和 `crash_or_body_contact`、`horizontal_position_outside_viewport`（失败）。当前代码有 `landing_quality` 组件促进 settled 状态，但完全没有 crash 或出界的前兆信号——没有惩罚横向位置边界接近、没有严重倾覆的预警。
- **目标→进度**: 主目标"安全准确降落并稳定停靠"被 `progress_reward` + `distance_shaping`（接近）和 `landing_quality`（接触 + 低运动）覆盖。但 `landing_quality` 只在接触门控打开时激活，而 agent 若长期不接触，则只有距离信号。
- **效率信号**: 动作维度 = 4（有离散动作），已含 `fuel_efficiency`——这其实是 `fuel_penalty`，但惩罚力度可能抑制了必要的主发动机使用。
- **僵尸组件**: `landing_quality` 的 active_rate 未知但大概率极低（需要接触 gate>0 且所有安全因子>0）；iter 6 的 `landing_bonus`（二值成功奖励）被替换为 iter 7 的 `landing_quality`（连续因子乘积），score 从 50.66 暴跌至 -37.65，len 从 703 升至 1000——乘积塌缩导致梯度消失。
- **一句话结论**: 当前 reward 漏了 crash/出界的前兆信号，且 `landing_quality` 乘积形式在接触前完全塌缩为 0，agent 缺乏靠近平台后的精调梯度。

## 行为诊断
- **agent 在做什么？** 所有 episode 都 truncated 在 1000 步，没有提前终止。agent 学会了避免 crash/出界（否则会有提前终止），但也没有完成 landing（否则会有 settled 提前终止）。它在"存活但不降落"——在视口内徘徊，利用 `distance_shaping` 和 `progress_reward` 获取正向信号，但因为缺乏接触后的明确奖励而无法完成最终步骤。这解释了 score=-37.65：距离信号提供的正值被 velocity_constraint、orientation_penalty、fuel_penalty 等累积惩罚所淹没。
- **干预目标**: 提供从"接近平台"到"接触降落"之间的连续梯度，让 agent 在接近时敢于并愿意进行接触。核心是修复 `landing_quality` 的接触前塌缩问题。
- **方向还值得继续吗？** 这是第一轮反思，iter 7 的 `landing_quality` 乘积形式已被证明劣于 iter 6 的 `landing_bonus`。不应继续修补乘积结构，而要替换为在接触前后都有梯度的形式。

## 干预层级：Level 2 — 结构变换

**证据**:
- `landing_quality` 乘积 treo 塌缩为 0 时 contact_gate=0（接触前），此时 landing 信号完全消失。
- iter 6 的二值 `landing_bonus` 反而得到 50.66 分，说明稀疏信号好于塌缩的连续信号。
- 需要一种**在接触前也有梯度**的接近-接触连续奖励。

**变换**: 乘积 proxy 经常塌缩 → 替换为 `joint_condition_proxy` 的几何平均形式（但需要在无接触时依然活跃）。更好的方案：引入一个**接近信号**（dense, 无门控），并保留接触后的 landing_quality，但将其改为几何平均而非裸乘积。

**具体改动**: 
- 删除原 `landing_quality`（乘积塌缩）
- 新增 `approach_and_land` 组件：使用 `exp(-distance)` 形式的接近奖励 + 接触后的几何平均安全因子
- 同时将 `fuel_penalty` 从常量惩罚改为**仅在非必要时惩罚**（增加效率，减少不必要的惩罚负担）

## 设计校准
- 主信号 per-step: `progress_reward + distance_shaping` ≈ 0~3.0 per step
- `approach_and_land` per-step 应 ≤ 主信号的 0.5x，设为 ~1.0
- `fuel_penalty` 简化为仅对主发动机和姿态发动机做轻度惩罚，per-step ≤ 0.05

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v3: landing_quality → approach_and_land (dense proximity + safe landing);
                fuel_efficiency → efficiency_gate (only penalize non-essential actions).
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

    # --- Component A: delta_distance (keep) ---
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (nx_pos**2 + ny_pos**2) ** 0.5
    delta_distance = current_distance - next_distance
    progress_reward = 10.0 * delta_distance

    # --- Component B: bounded_distance_shaping (keep) ---
    distance_shaping = 2.0 * (1.0 / (1.0 + 0.5 * next_distance))

    # --- Component C: velocity_hinge_constraint (keep) ---
    h_speed = abs(nx_vel)
    h_penalty = max(0.0, h_speed - 2.0)
    v_speed = ny_vel
    v_down_penalty = max(0.0, -v_speed - 1.5)
    v_up_penalty = max(0.0, v_speed - 1.0)
    angular_penalty = max(0.0, abs(n_angular_vel) - 0.8)
    velocity_constraint = -0.5 * (h_penalty + v_down_penalty + v_up_penalty + angular_penalty)

    # --- Component D: orientation_stabilization (gated, keep) ---
    contact_gate = (n_left_contact + n_right_contact) / 2.0
    orientation_penalty = (1.0 - contact_gate) * (-0.3 * (n_body_angle**2) - 0.2 * (n_angular_vel**2))

    # --- Component E: efficiency_gate (replacing fuel_penalty) ---
    # Penalize main engine and orientation engines only when agent is close to platform
    # and doesn't need aggressive thrust — promote efficient final approach.
    proximity_gate = max(0.0, 1.0 - next_distance / 1.5)  # 0 at dist>=1.5, 1 at dist=0
    if action == 0:
        efficiency_penalty = 0.0
    elif action == 2:
        efficiency_penalty = -0.08 * proximity_gate
    else:
        efficiency_penalty = -0.03 * proximity_gate

    # --- Component F: approach_and_land (replacing landing_quality) ---
    # Proximity factor: exponential decay, gives gradient even when far away
    proximity_factor = 2.718281828 ** (-2.0 * next_distance)  # 1 at dist=0, ~0.135 at dist=1
    
    # Safe landing factors (geometric mean, only active on contact to avoid noise)
    speed_norm = (nx_vel**2 + ny_vel**2) ** 0.5
    safe_speed_factor = max(0.0, 1.0 - speed_norm / 0.4)
    safe_angle_factor = max(0.0, 1.0 - abs(n_body_angle) / 0.2)
    safe_spin_factor = max(0.0, 1.0 - abs(n_angular_vel) / 0.2)
    
    # Geometric mean of safety factors (no collapse if one factor is near 0)
    safe_factors_product = safe_speed_factor * safe_angle_factor * safe_spin_factor
    safe_geo_mean = safe_factors_product ** (1.0 / 3.0) if safe_factors_product > 0 else 0.0
    
    # Blend: proximity gives continuous signal; when in contact, safety improves it
    approach_and_land = proximity_factor * (1.0 + 2.0 * contact_gate * safe_geo_mean)

    # --- Assemble total reward ---
    total_reward = (
        progress_reward +
        distance_shaping +
        velocity_constraint +
        orientation_penalty +
        efficiency_penalty +
        0.8 * approach_and_land
    )

    components = {
        'progress_reward': progress_reward,
        'distance_shaping': distance_shaping,
        'velocity_constraint': velocity_constraint,
        'orientation_penalty': orientation_penalty,
        'efficiency_penalty': efficiency_penalty,
        'approach_and_land': 0.8 * approach_and_land
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前 reward 漏了 crash/出界前兆信号，landing_quality 乘积在接触前塌缩为 0。
- **behavior**: agent 存活但不降落，所有 episode truncated 在 1000 步，利用距离信号徘徊而不完成着陆。
- **signal**: 缺乏从"接近平台"到"接触降落"的连续梯度，接触前的 landing 信号完全消失。
- **level**: Level 2
- **hypothesis**: 用指数接近因子 + 接触后几何平均安全因子替代塌缩乘积，让 agent 在接近时获得持续增强的信号，解决 iter 7 比 iter 6 差的问题。
- **risk**: proximity_factor 可能让 agent 满足于接近但不接触（悬停），0.8 系数和 safety_bonus 的 2x 乘子旨在提供接触的额外激励。若 agent 仍不接触，需增强接触奖励的相对权重。