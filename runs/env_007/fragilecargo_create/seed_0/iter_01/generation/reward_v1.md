# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- unpack signals ---
    # crate-to-dock offsets (normalized by half-width 5.0, half-height 4.0)
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    # crate velocity (world frame, normalized by 3.0)
    cvx = obs[8]
    cvy = obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # crate heading
    crate_cos = obs[10]
    crate_sin = obs[11]

    # cart forward speed (normalized by 3.0)
    cart_fwd = obs[4]

    # contact flag
    contact = obs[14]

    # --- A. crate_to_dock_progress: improvement_delta on distance ---
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new
    w_progress = 10.0
    r_progress = w_progress * progress

    # --- B. crate_settling_and_alignment: joint_condition_proxy ---
    # proximity factor: 1 when at dock, decays with distance
    prox = 1.0 / (1.0 + 8.0 * dist_new)
    # speed factor: 1 when still, decays with speed
    speed_factor = 1.0 / (1.0 + 20.0 * crate_speed)
    # alignment factor: crate heading aligned with dock axis (assume dock axis = world x)
    # |cos(heading)| close to 1 means aligned with x-axis
    align = abs(crate_cos)
    # geometric mean of three continuous factors
    settle = (prox * speed_factor * align) ** (1.0 / 3.0)
    w_settle = 3.0
    r_settle = w_settle * settle

    # --- C. fragile_impact_penalty: hinge on relative speed during contact ---
    # relative speed proxy: cart forward speed vs crate speed
    rel_speed = abs(cart_fwd - crate_speed)
    # hinge: only penalize when contact AND relative speed exceeds threshold
    impact_threshold = 0.15
    impact_excess = max(0.0, rel_speed - impact_threshold)
    w_impact = 5.0
    r_impact = -w_impact * impact_excess * contact

    # --- D. out_of_bounds_penalty: hinge on cart position near boundary ---
    cart_x = obs[0]
    cart_y = obs[1]
    # normalized bounds [-1, 1]; penalize when |pos| > 0.85
    bound_threshold = 0.85
    cart_x_excess = max(0.0, abs(cart_x) - bound_threshold)
    cart_y_excess = max(0.0, abs(cart_y) - bound_threshold)
    w_bounds = 8.0
    r_bounds = -w_bounds * (cart_x_excess + cart_y_excess)

    # --- total ---
    total_reward = r_progress + r_settle + r_impact + r_bounds

    components = {
        "crate_to_dock_progress": r_progress,
        "crate_settling_and_alignment": r_settle,
        "fragile_impact_penalty": r_impact,
        "out_of_bounds_penalty": r_bounds,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

**selected task_family / dynamics_subtype**
- task_family: `manipulation_grasping`
- dynamics_subtype: `goal_approach_and_soft_contact`

**selected reward roles**
1. `crate_to_dock_progress` (mandatory) — 主学习信号
2. `crate_settling_and_alignment` (mandatory) — 停稳与对齐近似
3. `fragile_impact_penalty` (conditional) — 易碎约束风险代理
4. `out_of_bounds_penalty` (conditional) — 越界约束

**role_to_signal_mapping**
| role | signals used | formula operator |
|---|---|---|
| crate_to_dock_progress | obs[12], obs[13], next_obs[12], next_obs[13] | `improvement_delta`（dist_old − dist_new），避免 proximity 悬停陷阱 |
| crate_settling_and_alignment | obs[8], obs[9], obs[10], obs[11], next_obs[12], next_obs[13] | `joint_condition_proxy`（几何平均），三个连续 factor：proximity、speed、alignment |
| fragile_impact_penalty | obs[14], obs[4], obs[8], obs[9] | `hinge_penalty`（仅当接触且相对速度超阈值时生效） |
| out_of_bounds_penalty | obs[0], obs[1] | `hinge_penalty`（仅当 |pos| > 0.85 时生效） |

**excluded roles 及原因**
- `action_smoothness`：任务未明确要求平滑/节能，v1 默认不加，避免抑制必要的推动动作。
- `time_penalty`：任务未明确要求速度，且可能诱导冒险行为增加硬冲击风险。
- `official_reward_shaping`：官方奖励被 mask，禁止使用。
- `info_based_success_bonus`：info 中 is_success / cargo_inside_dock / stable_steps 被禁止读取。
- `precise_hard_collision_penalty`：hard_collision_count 与 contact_impulse 被禁止读取，只能用接触+相对速度做风险代理。
- `cart_forward_velocity_reward`：小车速度本身不是目标，奖励会诱导乱冲。

**为什么没有使用 terminal_success_reward / terminal_failure_penalty**
- `explicit_success_flag_available: false`，info 中 is_success 被禁止读取，无法可靠检测成功终止。
- `explicit_failure_flag_available: false`，termination_reason / hard_collision_count 被禁止读取，无法可靠检测失败终止。
- 因此 v1 用 `crate_settling_and_alignment` 作为软完成近似（joint_condition_proxy），用 `fragile_impact_penalty` 和 `out_of_bounds_penalty` 作为前兆软信号。

**留到后续迭代的职责**
- action_smoothness（若观察到动作震荡）
- time_penalty（若观察到大量 truncation）
- 更精细的朝向对齐 shaping（若货箱朝向长期未对齐）
- 通过隔墙开口的路径引导（若货箱卡在开口）

**训练后应观察的 failure modes**
1. 货箱被推过 dock 无法停稳 → 检查 crate_speed 长期 > 0.05，obs[12..13] 震荡后远离
2. 硬冲击导致提前失败 → 检查 episode 突然终止前 obs[14]=1 且相对速度大
3. 货箱或小车越界 → 检查终止前 obs[0]/obs[1] 接近边界
4. 货箱朝向未对齐 → 检查货箱在 dock 内但 obs[10..11] 偏差大
5. 货箱卡在隔墙开口 → 检查货箱位置长期停滞在隔墙附近，obs[15..17] 接近 1
6. 时间耗尽 → 检查 obs[18] 接近 1
7. 主信号被悬停收割 → 检查货箱停在 dock 附近但未进入，reward 仍为正（已用 delta 缓解）
8. 动作震荡 → 检查 action[0]/action[1] 高频大幅变化