# 设计理由
从训练反馈看：全部 truncated（步长 1000）、无 terminated、得分 -11.79。这说明 agent 在“徘徊”而非“着陆”。当前代码存在一个塌缩的 `safe_landing_penalty`：当 agent 接近地面（`y<2.0`）时，`landing_gate` 升高，运动惩罚急剧放大，agent 选择远离地面以避免惩罚，于是永远悬停。需要一个直接鼓励“着陆状态”的组件，让接近 + 双接触 + 极低动能获得正向激励，同时削弱全局运动惩罚以避免地面恐惧。

本轮在 Level 2 做结构变换：把 `goal_proximity`（通用接近）替换为 **joint_condition_proxy**（联合条件连续 proxy），设计为：
- `touch_bonus = left_contact + right_contact`（0..2）
- `dist_factor = 1.0/(1.0 + dist_sq)`（接近 1 在原点）
- `speed_factor = 1.0/(1.0 + x_vel**2 + y_vel**2 + 0.1*ang_vel**2)`（动能极低时接近 1）
乘积 `touch_bonus * dist_factor * speed_factor` 形成着陆条件连续奖励，同时将原来的 `safe_landing_penalty` 系数从 -0.5 降至 -0.1 以缓解地面恐惧。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs signals
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- Component A: Landing proxy (replaces generic goal_proximity) ----
    # Touch factor: 0 when airborne, up to 2 when both legs touch
    touch_bonus = left_contact + right_contact
    # Distance factor: close to 1 when near pad center
    dist_sq = x_pos**2 + y_pos**2
    dist_factor = 1.0 / (1.0 + dist_sq)
    # Speed factor: close to 1 when near-zero velocity (stable)
    speed_sq = x_vel**2 + y_vel**2 + 0.1 * ang_vel**2
    speed_factor = 1.0 / (1.0 + speed_sq)
    # Joint reward: product of three bounded (0..1 or 0..2) factors
    landing_proxy = 2.0 * touch_bonus * dist_factor * speed_factor

    # ---- Component B: Safe landing constraint (soft gate, reduced) ----
    height_gate = max(0.0, 1.0 - y_pos / 2.0) if y_pos < 2.0 else 0.0
    contact_gate = 0.2 * (left_contact + right_contact)
    landing_gate = height_gate + contact_gate

    # Motion cost still penalises heavy movement near ground, but coefficient is now -0.1
    motion_cost = x_vel**2 + y_vel**2 + body_angle**2 + 0.1 * ang_vel**2
    safe_landing_penalty = -0.1 * landing_gate * motion_cost

    # ---- Component C: Fuel efficiency ----
    fuel_penalty = -0.01 if action != 0 else 0.0

    # ---- Total reward ----
    total_reward = landing_proxy + safe_landing_penalty + fuel_penalty

    components = {
        "landing_proxy": landing_proxy,
        "safe_landing_penalty": safe_landing_penalty,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少鼓励着陆的信号，且现有的 safe_landing_penalty 系数过高导致地面区域成为“死亡区”，致使 agent 完全不敢接近地面。
- **behavior**: agent 在 y>2 附近悬停徘徊，步长打满 1000，无终止无 crash，靠距离奖励维持正分，但永远不会着陆。
- **signal**: 需要直接对齐“稳定双足接触 + 接近中心 + 极低动能”的连续着陆奖励，削弱对运动惩罚的全局恐惧。
- **level**: Level 2
- **hypothesis**: 引入联合条件连续乘积 `touch_bonus * dist_factor * speed_factor` 给着陆状态明确正信号，降低 `safe_landing_penalty` 系数让 agent 敢于靠近地面，从而从徘徊转变为尝试着陆。
- **risk**: 着陆奖励的乘积可能在早期接触前塌缩为 0，导致梯度稀疏；需在后续迭代中观察 active_rate 是否维持。