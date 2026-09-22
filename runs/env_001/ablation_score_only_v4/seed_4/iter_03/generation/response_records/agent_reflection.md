# Response Record

# 设计理由

## 0. 信号覆盖审计
a) 终止→前兆：#5 §5 声明了 `body_not_awake_or_settled`(成功停靠) 和 `crash_or_body_contact`(碰撞) 两种终止。当前代码的 `landing_proxy` 通过 touch + dist + speed 给出了停靠前兆，但缺少对 crash 风险的软惩罚（如角度过大、速度过高时给予警告）。`safe_landing_penalty` 名义上做安全约束，但它惩罚的是 landing_gate × motion_cost，在gate≈0时（高度>2.0且无接触）完全不生效，这个设计是合理的——它只在地面附近才约束动作。

b) 目标→进度：#5 §1 声明目标是"到达并稳定停靠在中央目标着陆区域"，代码的 `landing_proxy` 包含 touch_bonus、dist_factor、speed_factor，理论上覆盖了进度信号。但 active_rate=98.25% 而 episode_sum_mean=-1.418 < 0 表明**该组件的符号或校准有问题**：一个叫 `landing_proxy` 的组件应该在接近目标时为正，但它每步平均为负。

c) 效率信号：#5 §4 动作维度=4 (< 6)，按规则不触发 action penalty 信号。但当前代码已有 fuel_penalty (-0.01 per action != 0)，且 active_rate 仅 0.46%，几乎无效——agent 没有被引导节省燃料。

d) 僵尸组件：fuel_penalty active_rate=0.46%（< 2%），是僵尸组件，应考虑删除或重构。

e) 一句话结论：`safe_landing_penalty` (active_rate=63%, episode_sum_mean=-44.3) 是奖励崩塌的主因——它是唯一大量触发且为负的组件，每步平均贡献 -0.07，远超 landing_proxy 的量级；同时 `landing_proxy` 本身期望为正却为负，可能因为 `touch_bonus` 在空中为 0，导致乘积整体塌缩为 0（甚至数值极小），而 agent 长期无法触地也得不到梯度。

## 1. 行为诊断
1. **agent 在做什么？** 所有 episode truncated=20/20，len=1000，terminated=0，说明 agent 在避免失败的同时一直无法成功着陆。它在空中慢速徘徊（len 满值但未 crash），但 `landing_proxy` 的负均值说明它没有得到正确的梯度引导下降。

2. **干预哪个目标？** 需要修复两类问题：(1) `safe_landing_penalty` 过大主导了奖励，使得 agent 对运动的恐惧压倒了下降动机；(2) `landing_proxy` 在空中时 touch_bonus=0，乘积为 0，agent 无梯度可循。应同时削弱惩罚并重构着陆引导信号。

3. **方向还值得继续吗？** 从 iter 1→2 看，当 `safe_landing_penalty` 系数从原始值降为 -0.1 后 score 反而从 -11.79 暴跌到 -37.08。因为其 active_rate=63% 说明它仍在大量触发，-0.1 仍然过高。需要更大幅度的削弱才能改变主导权。

## 2. 干预层级: Level 2 — 结构变换
- `safe_landing_penalty` 虽以 gate 形式存在，但 gate 值在某些区间过强，且 `landing_gate` 中的 `contact_gate` 只乘 0.2，当双足同时触地时 gate=1.4，乘 motion_cost 后惩罚巨大。需要将其降低到次要地位。
- `fuel_penalty` active_rate < 2%，触发率极低，直接删除（僵尸组件）。
- `landing_proxy` 在空中时塌缩为 0 或负值，需要重构为**在空中也有梯度**的形式。

## 3. 设计校准
- 主信号 per-step ≈ 0.002（landing_proxy 每步 0.002 的正期望即可引导）
- 新 landing_proxy 应给出明确的正导向：离目标越近（x,y 小）、速度越慢、角度越正，奖励越高。
- 惩罚总负担 ≤ 主信号的 0.5x：safe_landing_penalty per-step ≤ 0.001。

## 4. 新设计
**修改 landing_proxy 为铰链奖励（hinged dense reward）**，在空中也提供正梯度：
- 位置分：`1.0 / (1.0 + dist)` — 距离越近、值越大，始终为正
- 速度分：`1.0 / (1.0 + |v|)` — 低速奖励
- 角度分：`1.0 - |body_angle|/π` （有界 [0,1]）— 竖直奖励
- 接触分：`0.3 * (left_contact + right_contact)` — 双足触地有额外奖励

**削弱 safe_landing_penalty**：系数从 -0.1 → -0.01（降低 10×），使其成为微调项而非主导项。

**删除 fuel_penalty**（僵尸组件，active_rate < 2%）。

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

    # ---- Component A: Dense landing progress (hinged, always positive gradient) ----
    # Position: close to pad center
    dist = (x_pos**2 + y_pos**2) ** 0.5
    position_factor = 1.0 / (1.0 + dist)  # [0,1], decreasing with distance
    
    # Velocity: prefer low speed
    speed = (x_vel**2 + y_vel**2 + 0.1 * ang_vel**2) ** 0.5
    speed_factor = 1.0 / (1.0 + speed)  # [0,1], decreasing with speed
    
    # Orientation: prefer upright
    angle_factor = max(0.0, 1.0 - abs(body_angle) / 3.14159265)  # [0,1], 1=perfectly upright
    
    # Contact: bonus for legs on ground
    contact_bonus = 0.3 * (left_contact + right_contact)  # [0, 0.6]
    
    # Combined landing proxy: weighted sum with contact as additive bonus
    landing_proxy = 1.5 * position_factor * speed_factor * angle_factor + contact_bonus

    # ---- Component B: Safe landing penalty (greatly reduced) ----
    # Gate: active only when near ground or in contact
    height_gate = max(0.0, 1.0 - y_pos / 2.0) if y_pos < 2.0 else 0.0
    contact_gate = 0.2 * (left_contact + right_contact)
    landing_gate = height_gate + contact_gate
    
    # Motion cost near ground
    motion_cost = x_vel**2 + y_vel**2 + body_angle**2 + 0.1 * ang_vel**2
    safe_landing_penalty = -0.01 * landing_gate * motion_cost

    # ---- Total reward ----
    total_reward = landing_proxy + safe_landing_penalty

    components = {
        "landing_proxy": landing_proxy,
        "safe_landing_penalty": safe_landing_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少 crash 前兆信号，僵尸组件 fuel_penalty 无作为，safe_landing_penalty 主导奖励使其严重为负
- **behavior**: agent 在空中慢速徘徊，所有 episode 跑满 1000 步无法着陆
- **signal**: safe_landing_penalty 过强（主导负奖励），landing_proxy 在空中塌缩为 0 无梯度
- **level**: Level 2
- **hypothesis**: 将 landing_proxy 重构为始终提供正梯度的密集奖励 + 将 safe_landing_penalty 降至 1/10，应使 agent 获得明确的下降动机，同时不再因过强的运动恐惧而失去探索
- **risk**: 若速度分量权重过大，agent 可能在远端就过度减速，延缓任务完成；但 speed_factor 用 1/(1+speed) 是平滑的，不应极端克速
