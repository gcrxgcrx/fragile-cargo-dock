# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v1 for 2D Lander goal reaching with soft landing.
    
    Design hypothesis: The previous attempts used proximity + contact bonus + landing penalty
    and stagnated at ~5.7. The hypothesis is that contact bonus alone was sparse and possibly
    exploited, while the proximity reward lacked velocity-conditional gating and orientation
    coupling. This version shifts to:
    - delta_distance as the primary progress driver (dense, per-step gradient)
    - bounded absolute distance as a shaping complement (prevents drifting)
    - velocity+angular constraints via hinge penalties (only when dangerous)
    - orientation+angular velocity quadratic suppression (continuous guidance)
    - minimal fuel penalty (no conditional gating, just light discouragement)
    
    This materially differs from the tried structure by:
    1) Using improvement_delta as the main progress signal instead of absolute proximity
    2) Removing discrete contact bonus entirely (replaced by implicit soft-landing shaping)
    3) Using hinge penalties for velocity instead of quadratic (gives freedom in safe range)
    """
    
    # --- Unpack observations ---
    # Current state
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    
    # Next state
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_body_angle = next_obs[4]
    n_angular_vel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]
    
    # --- Component A: delta_distance (primary progress driver) ---
    # Encourage reducing distance to target pad center (0, 0)
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (nx_pos**2 + ny_pos**2) ** 0.5
    delta_distance = current_distance - next_distance
    
    # Scale: small positive reward for approaching, near-zero when stationary
    progress_reward = 10.0 * delta_distance
    
    # --- Component B: bounded_distance_shaping (absolute proximity guidance) ---
    # Complement to delta: ensures agent doesn't drift far even if delta is zero
    # Use 1/(1+k*d) to give strong gradient near target, saturates at large distances
    distance_shaping = 2.0 * (1.0 / (1.0 + 0.5 * next_distance))
    
    # --- Component C: velocity_hinge_constraint (soft safety on speed) ---
    # Penalize horizontal speed when too high (over 2.0 m/s)
    h_speed = abs(nx_vel)
    h_penalty = max(0.0, h_speed - 2.0)
    
    # Penalize vertical speed: downward too fast (> -1.5, penalize magnitude),
    # upward too fast (> 1.0, penalize)
    v_speed = ny_vel
    v_down_penalty = max(0.0, -v_speed - 1.5)  # penalize when more negative than -1.5
    v_up_penalty = max(0.0, v_speed - 1.0)     # penalize when more positive than 1.0
    
    # Angular velocity penalty: penalize when |angular_vel| > 0.8
    angular_penalty = max(0.0, abs(n_angular_vel) - 0.8)
    
    velocity_constraint = -0.5 * (h_penalty + v_down_penalty + v_up_penalty + angular_penalty)
    
    # --- Component D: orientation_stabilization (quadratic on angle + angular vel) ---
    # Continuous penalty for tilt and rotation, light enough to not freeze the agent
    orientation_penalty = -0.3 * (n_body_angle**2) - 0.2 * (n_angular_vel**2)
    
    # --- Component E: fuel_efficiency (light action penalty) ---
    # Encourage no_engine (action 0) when possible
    # action 0: no_engine, 1: left_orientation, 2: main_engine, 3: right_orientation
    if action == 0:
        fuel_penalty = 0.0
    elif action == 2:
        fuel_penalty = -0.08  # main engine: slightly more expensive
    else:
        fuel_penalty = -0.05  # orientation engines: light penalty
    
    # --- Assemble total reward ---
    total_reward = (
        progress_reward +
        distance_shaping +
        velocity_constraint +
        orientation_penalty +
        fuel_penalty
    )
    
    components = {
        'progress_reward': progress_reward,
        'distance_shaping': distance_shaping,
        'velocity_constraint': velocity_constraint,
        'orientation_penalty': orientation_penalty,
        'fuel_penalty': fuel_penalty
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **task_family:** `navigation_goal_reaching`
- **dynamics_subtype:** `goal_approach_and_soft_contact`
- 核心挑战：引导飞行器从高处向目标垫移动，同时控制速度、姿态，实现平稳接触，不依赖显式的成功标志。

## selected reward roles
按 environment_card.md `reward_role_decomposition` 选择：

1. **`distance_to_target`（mandatory）** — 主学习信号
   - 使用 `improvement_delta` 和 `bounded_signal` 组合。
   - 组件 A（`progress_reward`）：每一步距离减少即获得正奖励，提供稠密梯度。
   - 组件 B（`distance_shaping`）：绝对距离的平滑压缩，确保 agent 不因 delta 为零而漂移。

2. **`soft_landing_velocity`（mandatory）** — 安全约束
   - 使用 `dense_state_signal` 的 hinge 形式。
   - 仅当线速度或角速度超出安全区间时施加惩罚，避免在安全范围内抑制正常运动。

3. **`upright_orientation`（mandatory）** — 姿态稳定
   - 使用 `quadratic_penalty` 对 body_angle 和 angular_vel 进行连续轻量抑制。

4. **`fuel_penalty`（conditional）** — 效率约束
   - 线性惩罚非零动作，权重极小（0.05–0.08），确保不阻碍早期探索。

## role_to_signal_mapping
| role | signals used | formula operator |
|---|---|---|
| `distance_to_target` | `next_obs[0]`, `next_obs[1]`（用于 next_distance）; `obs[0]`, `obs[1]`（用于 current_distance） | `improvement_delta`（progress_reward）; `bounded_signal` 倒数衰减（distance_shaping） |
| `soft_landing_velocity` | `next_obs[2]`, `next_obs[3]`, `next_obs[5]` | `dense_state_signal` hinge penalty |
| `upright_orientation` | `next_obs[4]`, `next_obs[5]` | `quadratic_penalty` |
| `fuel_penalty` | `action` | 线性惩罚（按动作类型区分） |

## excluded roles 及原因
- **`contact_reward`**：之前尝试的 contact bonus 可能导致 reward hacking（猛烈撞击以获取接触奖励），且稀疏。v1 通过速度/角度约束隐式引导平稳接触，留到后续迭代再考虑是否需要 safety-gated contact bonus。
- **`time_step_penalty`**：缺少全局步数信号。
- **`success_oriented_bonus`**：`explicit_success_flag_available=false`。
- **`exact_zoning_reward`**：无区域划分信号。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- `explicit_success_flag_available=false`，`explicit_failure_flag_available=false`。
- info 为空字典，无法获取终止原因。
- 所有引导必须依赖于每步可观测的连续信号，而非终止事件。

## 哪些职责留到后续迭代
- **gated velocity penalty**（在近距离时自动增强）：当前使用固定阈值的 hinge，未与距离耦合。若出现“接近目标时仍高速”的问题，可引入 `velocity_constraint * (1 - distance_to_target / threshold)` 形式的门控。
- **contact stability reward**：若 agent 学会接触但不稳定（弹跳或倾斜），可引入 leg contact 与低速度/小角度的联合条件 proxy。
- **dynamic curriculum**（早期容忍高速度，后期严格要求）：当前未使用 `training_progress`，若 early/late 阶段存在明显冲突再引入。

## 训练后应观察的 failure modes
- **hover_near_target**：若 progress_reward 过弱，agent 可能在目标附近徘徊而不着陆。观察 terminal distance 和 velocity。
- **velocity_cap_exploit**：agent 在 hinge 阈值内以最大速度运动，导致接触时仍过快。观察 contact-frame velocity。
- **orientation_stall**：agent 为保持竖直而拒绝旋转，stuck 在错误水平位置。观察 success rate 和 terminal error。
- **fuel_penalty_freezing**：极轻的燃料惩罚可能在后期被忽略，可以后续提高或改为门槛式惩罚。
- **delta_distance_oscillation**：agent 左右振荡以持续获得 progress delta。观察 trajectory smoothness。