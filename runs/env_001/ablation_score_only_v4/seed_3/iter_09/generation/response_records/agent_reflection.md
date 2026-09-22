# Response Record

# 设计理由

## 信号覆盖审计

**a) 终止 → 前兆**：环境声明了三个终止条件：settled（成功）、crash（失败）、out_of_viewport（失败）。当前代码对 crash 无直接前兆预警（仅是 velocity_constraint 的间接惩罚），对 settled 无实质性前兆引导——`approach_and_land` 的安全因子仅在 `contact_gate > 0`（即已接触）时才激活 `safe_geo_mean`，但 settled 要求的是机体停稳（低速度+低角速度+低角度），需要这些条件在接触前就开始建立。

**b) 目标 → 进度**：主目标是安全着陆+稳定停靠。当前奖励的最大信号来自 `progress_reward`（推动 xy 趋近），`approach_and_land` 的 proximity_factor 确实在推动接近，但"安全着陆质量"信号在非接触状态完全失效 `(contact_gate=0 → safe_geo_mean 被乘 0)`。

**c) 效率信号**：action_space n=4 ≥ 3，已包含 efficiency_penalty，覆盖完备。

**d) 僵尸组件**：从 training_summary 组件表可推断 `orientation_penalty` 和 `efficiency_penalty` 的 active_rate 可能极低——前者被 contact_gate 门控（仅在接触时活跃），后者被 proximity_gate 门控（仅在 next_distance < 1.5 时活跃）。需要检查数据确认，但结构性上已存在低触发风险。

**e) 一句话结论**：当前 reward 缺少"接触前着陆准备质量"的连续梯度信号——agent 在接近平台时得不到关于"应该减速、摆正、稳角速度"的任何引导，只能在碰触后才知道做得好不好，而碰触瞬间若状态不佳直接导致 crash 或弹飞。

## 行为诊断

**agent 在做什么？** 所有 20 个 episode 均 truncated（len=1000），说明 agent 既不 crash 也不 settle——它在长时间徘徊。score_range 从 -76 到 4，说明有的 episode 在接近平台但无法成功着陆。这符合"缺少着陆前准备信号"的预期：agent 能学会接近平台（proximity 有梯度），但无法学会在接近时减速摆正（安全因子仅在接触后激活）。

**干预哪个目标？** 修改 `approach_and_land` 组件，将"安全着陆质量"信号从"仅接触后激活"改为"接近阶段连续激活"，让 agent 在接近平台的过程中就能获得调整姿态、角速度、速度的梯度。

**这个方向还值得继续吗？** 这是第一轮诊断（#3 空），当前骨架（approach_and_land + progress + velocity_constraint）在 iter 6 曾达到 50.66 分，说明方向有潜力。但安全着陆信号的缺失是结构性缺陷，需要修补。

## 干预层级：Level 2 — 结构变换

**证据**：`safe_geo_mean` 仅在 `contact_gate` 为非零时有效（== 已接触），而接触前 proximity_factor 只是一个无差别的接近奖励（不区分"以安全姿态接近"还是"以危险姿态接近"）。这属于"占据好状态即持续获奖"的反模式——proximity_factor 给了撞向平台和优雅降落同样多的奖励。

**变换**：将 `approach_and_land` 升级为 `safe_approach`，使用几何平均形式构建一个连续的安全接近信号，在接近阶段就激活速度、角度、角速度的安全约束。几何平均保证任一维度极差时整体信号仍平滑下降（不完全塌缩），且在所有距离都有梯度。

**系数校准**：
- 主信号 per-step：progress_reward 约 10 * delta_distance，每步约 0.02-0.5（从 score/len 推断）。新 safe_approach 应设计为 per-step 约 0.01-0.15，在接近时提供有意义的梯度但不压倒前进动力。
- 总惩罚负担：velocity_constraint + efficiency_penalty + orientation_penalty ≈ 0-0.1 per step，合理。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v4: safe_approach (landing readiness throughout final descent) replaces
                approach_and_land (which only activated safety on contact).
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

    # --- Component F: safe_approach (replaces approach_and_land) ---
    # Continuous landing-readiness signal that activates as distance decreases.
    # Uses geometric mean of three safety factors, blended with proximity.
    
    # Proximity weight: ramps from 0 at dist >= 5 to 1 at dist = 0
    proximity_weight = max(0.0, 1.0 - next_distance / 5.0)
    
    # Safety factors (bounded [0, 1], soft ramp from "dangerous" to "safe")
    speed_norm = (nx_vel**2 + ny_vel**2) ** 0.5
    # Safe speed: target < 0.5, dangerous > 1.5
    safe_speed = max(0.0, 1.0 - speed_norm / 1.5)
    # Safe angle: target < 0.15, dangerous > 0.5
    safe_angle = max(0.0, 1.0 - abs(n_body_angle) / 0.5)
    # Safe spin: target < 0.15, dangerous > 0.5
    safe_spin = max(0.0, 1.0 - abs(n_angular_vel) / 0.5)
    
    # Geometric mean: prevents total collapse when one factor is slightly off
    safe_product = safe_speed * safe_angle * safe_spin
    safe_geo_mean = safe_product ** (1.0 / 3.0) if safe_product > 0.0 else 0.0
    
    # Safe approach: proximity_weight gates the landing-readiness signal
    # When far away, agent focuses on approaching; as it gets close, safety matters.
    safe_approach = proximity_weight * safe_geo_mean * 2.0

    # --- Assemble total reward ---
    total_reward = (
        progress_reward +
        distance_shaping +
        velocity_constraint +
        orientation_penalty +
        efficiency_penalty +
        safe_approach
    )

    components = {
        'progress_reward': progress_reward,
        'distance_shaping': distance_shaping,
        'velocity_constraint': velocity_constraint,
        'orientation_penalty': orientation_penalty,
        'efficiency_penalty': efficiency_penalty,
        'safe_approach': safe_approach
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺失"接触前着陆准备质量"的连续梯度——安全信号仅在已接触后激活，agent 在接近阶段无法学习减速/摆正。
- **behavior**: agent 在 long-horizon 徘徊（truncated=20/20, len=1000），能接近但无法安全着陆——因为不知道接近时该怎么做。
- **signal**: `approach_and_land` 的 safe_geo_mean 被 contact_gate 完全门控，接触前恒为 0；需替换为 proximity-weighted 连续安全信号。
- **level**: Level 2 — 结构变换（将仅接触后激活的安全信号改为接近阶段全程激活的连续 bounded factor + 几何平均）。
- **hypothesis**: safe_approach 提供 proximity 衰减的着陆准备梯度，agent 在接近平台时可学习减速、降低角速度、摆正姿态，从而将接近转化为着陆。
- **risk**: proximity_weight=0 的远处区域，safe_approach=0，agent 不再有"保持姿态好"的激励，可能在远处就姿态恶化。但 progress_reward + velocity_constraint 足够维持基本秩序；若下一轮观察到远处姿态恶化，可加入弱全局姿态约束。
