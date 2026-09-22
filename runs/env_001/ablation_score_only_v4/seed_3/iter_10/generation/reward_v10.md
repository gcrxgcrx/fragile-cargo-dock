# 设计理由
当前 reward 的 `safe_approach` 组件在接近目标时持续给奖（无论是否接触），导致 agent 学会在平台上方以相对安全的姿态盘旋，却缺乏完成最后 touchdown 的激励。历史迭代也显示，每次移除“着陆奖励”后分数明显下降（iter 6 有 landing_bonus 得分 50.66，后续替换后跌至负值）。因此，本轮的改动是把 `safe_approach` 重构为 **`landing_bonus`**：仅在支撑腿接触时给予与安全程度成正比的奖励，强激励 agent 完成从“接近”到“着陆”的最后一跳。

**数学形式**  
- 保留安全因子 `safe_speed`、`safe_angle`、`safe_spin` 的 soft ramp 几何平均 `safe_geo_mean`。  
- 引入 `contact_gate = (n_left_contact + n_right_contact) / 2`，作为着陆进度（0/0.5/1）。  
- `landing_bonus = contact_gate * safe_geo_mean * 1.5` —— 仅当有接触时才给出正奖励，且完全安全时每步 +1.5。  
- 接近过程的引导继续由 `progress_reward`、`distance_shaping`、`velocity_constraint` 和 `orientation_penalty` 承担，因此删除 `proximity_weight` 是安全的。

**系数校准**  
- `1.5` 约为当前主信号 per‑step（`progress_reward` 后期约 0.2~0.5 + `distance_shaping` ~0.5）的 1~2 倍，属于可接受范围。  
- 无新惩罚，不增加负担。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v5: landing_bonus replaces safe_approach — contact-gated safety reward
                to incentivize actual touchdown instead of hovering nearby.
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

    # --- Component A: delta_distance (unchanged) ---
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (nx_pos**2 + ny_pos**2) ** 0.5
    delta_distance = current_distance - next_distance
    progress_reward = 10.0 * delta_distance

    # --- Component B: bounded_distance_shaping (unchanged) ---
    distance_shaping = 2.0 * (1.0 / (1.0 + 0.5 * next_distance))

    # --- Component C: velocity_hinge_constraint (unchanged) ---
    h_speed = abs(nx_vel)
    h_penalty = max(0.0, h_speed - 2.0)
    v_speed = ny_vel
    v_down_penalty = max(0.0, -v_speed - 1.5)
    v_up_penalty = max(0.0, v_speed - 1.0)
    angular_penalty = max(0.0, abs(n_angular_vel) - 0.8)
    velocity_constraint = -0.5 * (h_penalty + v_down_penalty + v_up_penalty + angular_penalty)

    # --- Component D: orientation_stabilization (gated, unchanged) ---
    contact_gate = (n_left_contact + n_right_contact) / 2.0
    orientation_penalty = (1.0 - contact_gate) * (-0.3 * (n_body_angle**2) - 0.2 * (n_angular_vel**2))

    # --- Component E: efficiency_gate (unchanged) ---
    proximity_gate = max(0.0, 1.0 - next_distance / 1.5)
    if action == 0:
        efficiency_penalty = 0.0
    elif action == 2:
        efficiency_penalty = -0.08 * proximity_gate
    else:
        efficiency_penalty = -0.03 * proximity_gate

    # --- Component F: landing_bonus (replaces safe_approach) ---
    # Contact-gated safety reward: only gives positive reward when legs touch.
    # Uses the same soft safety factors, kept in geometric mean to avoid collapse.
    speed_norm = (nx_vel**2 + ny_vel**2) ** 0.5
    safe_speed = max(0.0, 1.0 - speed_norm / 1.5)
    safe_angle = max(0.0, 1.0 - abs(n_body_angle) / 0.5)
    safe_spin  = max(0.0, 1.0 - abs(n_angular_vel) / 0.5)
    safe_product = safe_speed * safe_angle * safe_spin
    safe_geo_mean = safe_product ** (1.0 / 3.0) if safe_product > 0.0 else 0.0

    landing_bonus = contact_gate * safe_geo_mean * 1.5

    # --- Assemble total reward ---
    total_reward = (
        progress_reward +
        distance_shaping +
        velocity_constraint +
        orientation_penalty +
        efficiency_penalty +
        landing_bonus
    )

    components = {
        'progress_reward': progress_reward,
        'distance_shaping': distance_shaping,
        'velocity_constraint': velocity_constraint,
        'orientation_penalty': orientation_penalty,
        'efficiency_penalty': efficiency_penalty,
        'landing_bonus': landing_bonus
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少水平出界前兆信号（当前存活率尚可，暂不致命），但更致命的是缺少着陆接触的正激励，导致 agent 只接近不 touchdown。
- **behavior**: agent 学会在目标附近以较安全姿态徘徊（len≈965，大多被截断而非成功着陆），未能触发成功终止。
- **signal**: 缺少“接触 + 安全”的奖励；原 `safe_approach` 在无接触时也给奖，削弱了着陆动力。
- **level**: Level 2
- **hypothesis**: 将奖励仅开放给接触状态后，agent 必须通过双腿接触并保持低速度/小角度来获取正奖励，从而完成从“接近”到“着陆”的过渡，提升成功终止率。
- **risk**: 若速度/角度约束不够强，agent 可能尝试高速撞击以快速获得接触奖励；但现有的 `velocity_constraint` 和 `orientation_penalty` 以及 `safe_geo_mean` 中对高速大角度的惩罚会抑制该行为。