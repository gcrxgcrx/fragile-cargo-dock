# 设计理由
当前奖励函数已驱动 agent 在第 4 轮取得 261.39 分、平均 325 步、19/20 成功着陆的优秀表现，主要任务（安全停靠）已超额完成。然而，次要目标“节省引擎推力”尚未被显式编码——`action` 信息未进入奖励，agent 可能在着陆后或接近目标时仍频繁使用引擎，导致不必要的燃料消耗。早期尝试 `fuel_penalty`（ir 1）因惩罚过重导致徘徊，但现在我们已经有一个行之有效的 `potential_gain` 主驱动，可以在不破坏现有策略的前提下，以极轻量级的方式加入 **action‑efficiency cost**，鼓励 `no_engine`（action 0）的选择，进一步贴合次要目标。

同时保持原有 `potential_gain` 和 `safe_landing_penalty` 不变。新组件 `action_efficiency_cost`：
- 形式：`-0.01 * (action != 0)`，即对任何使用引擎的动作（1/2/3）施加固定小额惩罚。
- 系数校准：`potential_gain` 的每步平均奖励约为 0.8（总分 261 / 步数 325 的一半左右，因为 `safe_landing_penalty` 占比较小），新惩罚 0.01 仅占其 1.25%，远低于 30% 阈值；加上原有 `safe_landing_penalty` 的每步惩罚约 0.0x 量级，总惩罚负担 ≤ 0.5x 主信号，安全。
- 不影响终止条件前兆或关键状态，但能提供每步的稀疏效率信号。

# 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current observation
    x_cur, y_cur = obs[0], obs[1]
    xv_cur, yv_cur = obs[2], obs[3]
    angle_cur = obs[4]
    angv_cur = obs[5]
    lc_cur, rc_cur = obs[6], obs[7]

    # Unpack next observation
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    xv_nxt, yv_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angv_nxt = next_obs[5]
    lc_nxt, rc_nxt = next_obs[6], next_obs[7]

    # ---- Potential function ----
    def potential(x, y, xv, yv, angle, angv, lc, rc):
        dist = (x**2 + y**2) ** 0.5
        pos_factor = 1.0 / (1.0 + dist)
        speed = (xv**2 + yv**2 + 0.1 * angv**2) ** 0.5
        speed_factor = 1.0 / (1.0 + speed)
        angle_factor = max(0.0, 1.0 - abs(angle) / 3.14159265)
        contact_factor = (lc + rc) / 2.0   # [0,1]
        return pos_factor * speed_factor * angle_factor * contact_factor

    pot_curr = potential(x_cur, y_cur, xv_cur, yv_cur, angle_cur, angv_cur, lc_cur, rc_cur)
    pot_next = potential(x_nxt, y_nxt, xv_nxt, yv_nxt, angle_nxt, angv_nxt, lc_nxt, rc_nxt)

    potential_gain = 2.0 * (pot_next - pot_curr)

    # ---- Component B: Safe landing penalty ----
    y_pos = next_obs[1]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]

    height_gate = max(0.0, 1.0 - y_pos / 2.0) if y_pos < 2.0 else 0.0
    contact_gate = 0.2 * (left_contact + right_contact)
    landing_gate = height_gate + contact_gate

    motion_cost = x_vel**2 + y_vel**2 + body_angle**2 + 0.1 * ang_vel**2
    safe_landing_penalty = -0.01 * landing_gate * motion_cost

    # ---- Component C: Action efficiency cost (new) ----
    # Encourage no_engine (action=0) to save fuel; tiny penalty for using any engine.
    action_efficiency_cost = -0.01 * (1.0 if action != 0 else 0.0)

    # ---- Total reward ----
    total_reward = potential_gain + safe_landing_penalty + action_efficiency_cost

    components = {
        "potential_gain": potential_gain,
        "safe_landing_penalty": safe_landing_penalty,
        "action_efficiency_cost": action_efficiency_cost
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 当前奖励缺少燃料效率信号和崩溃预防前兆，但主任务已充分解决；新增轻量 action 效率惩罚以修补次要目标缺口。
- **behavior**: agent 成功快速着陆（avg 325步，19/20 terminated），没有刷分或徘徊。
- **signal**: 缺少对引擎使用的节省信号，可能导致无谓推进。
- **level**: Level 2（添加新组件 `action_efficiency_cost`）
- **hypothesis**: 极小的 action 使用惩罚将让 agent 在已有成功策略基础上进一步减少无用引擎操作，提升燃料效率，与任务次要目标一致，且不影响着陆成功。
- **risk**: 系数极小，对主学习驱动干扰可忽略，但若早期训练中 `potential_gain` 信号尚弱时可能延缓探索；当前策略已稳定，风险很低。