# Response Record

# 设计理由

## 0. 信号覆盖审计

a) **终止→前兆**：环境声明了两个失败终止条件——`crash_or_body_contact`（撞击地面）和 `horizontal_position_outside_viewport`（x 越界）。当前代码没有任何针对"接近地面时垂直/水平速度过大"的 hinge 预警，也没有对 x 越界的软约束。agent 只有在撞毁或出界后才得到终止信号，缺乏前兆梯度。

b) **目标→进度**：`proximity_reward` 提供了向目标靠近的梯度（方向正确），但 `safe_contact_bonus` 是稀疏二值终点 bonus，无法在飞行过程中给 agent 任何"如何安全着陆"的连续引导。

c) **效率信号**：动作维度=4，未触发 action penalty 条件，跳过。

d) **僵尸组件**：`safe_contact_bonus` 需要双腿同时接触且 6 个状态因子全在容忍范围内才触发。在 agent 尚未学会着陆的阶段，该组件几乎不可能触发（active_rate 必然接近 0），对学习无实质贡献，是事实上的僵尸。本轮不动它，留作后续改造。

e) **一句话结论**：当前 reward 漏了"接近地面时对危险速度/姿态的软约束"信号，且已有的 `soft_landing_penalty` 惩罚形态不佳——对安全和不安全动态一视同仁地惩罚，缺乏阈值分界，无法给出"减速到安全范围"的明确梯度。

## 1. 行为诊断

- **agent 在做什么**：快速失败。全部 20 个 eval episode 均为 early terminal，len≈68 步即坠毁，score 在 -125 到 -95 之间。agent 从高空下降接近地面后因速度/姿态失控而撞毁。
- **负分根源**：`soft_landing_penalty` 在靠近地面（|y_err|<2.0）时激活，惩罚所有动态（无论是否安全）。agent 接地前 vy 通常较大，dynamics_cost 可达 2–4，每步惩罚约 -1～-2，而 `proximity_reward` 每步平均仅 +0.06。惩罚超出主信号 15–30 倍，严重压制了接近目标的正反馈。
- **干预目标**：修复 `soft_landing_penalty`，使其只惩罚"超出安全阈值"的过剩动态，而非对所有运动施加均匀惩罚。这同时解决了审计缺口（a）中缺失的"危险速度前兆信号"。

- **方向还值得继续吗**：这是该骨架的第 3 轮迭代，iter 1 曾拿到 +5.67 的正分（说明骨架尚存潜力），但 iter 2/3 崩塌。问题不在骨架结构，而在惩罚组件的数学形态塌缩——无阈值连续惩罚无法区分安全/危险。修一个组件即可。

## 2. 干预层级：Level 2 结构变换

**证据**：`soft_landing_penalty` 对安全和不安全动态施加无差别惩罚，且 per-step 惩罚量是主信号的 15–30x。这不是系数问题（即使降系数，无阈值惩罚依然无法给出"减速至此即可"的饱和信号），而是数学形态错误。

**变换**：将连续二次型惩罚 → hinge 惩罚（只惩罚超出安全阈值的部分）。

| 变换要素 | 旧形态 | 新形态 |
|---|---|---|
| vy 惩罚 | `-0.5 * gate * vy²/(1+vy²)` | `-0.05 * gate * max(0, \|vy\| - 0.5)` |
| vx 惩罚 | `-0.5 * gate * vx²/(1+vx²)` | `-0.05 * gate * max(0, \|vx\| - 0.5)` |
| angle 惩罚 | `-0.5 * gate * angle²/(1+angle²)` | `-0.05 * gate * max(0, \|angle\| - 0.2)` |
| angvel 惩罚 | `-0.5 * gate * angvel²/(1+angvel²)` | `-0.05 * gate * max(0, \|ang_vel\| - 0.2)` |

- **gate** 保留：`max(0, 1 - |y_err| / 2.0)`，确保只在接近地面时施加。
- **安全阈值**：vy/vx ≤ 0.5, angle ≤ 0.2, angvel ≤ 0.2 时无惩罚。这些值对应软着陆的安全包线。
- **系数校准**：0.05。在 gate=0.5、各项超出阈值合计约 1.0 的典型场景下，per-step 惩罚约 -0.025，相对 proximity_reward per-step (0.06–0.17) 的比例 ≤ 0.4x，满足设计校准要求。安全状态下惩罚严格为 0。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack previous and next observation
    x_err_prev = obs[0]
    y_err_prev = obs[1]
    dist_prev = (x_err_prev**2 + y_err_prev**2)**0.5

    x_err = next_obs[0]
    y_err = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    dist = (x_err**2 + y_err**2)**0.5

    # ---------- Role 1: proximity_to_target (improvement reward) ----------
    # Transform to improvement over step: reward approaching, penalise moving away
    f_prev = dist_prev / (1.0 + dist_prev)
    f_next = dist / (1.0 + dist)
    proximity_reward = -5.0 * (f_next - f_prev)

    # ---------- Role 2: soft_landing_dynamics (stability constraint) ----------
    # Hinge penalty: only penalise dynamics that exceed safe thresholds near ground
    landing_threshold = 2.0
    gate = max(0.0, 1.0 - abs(y_err) / landing_threshold)

    vy_safe = 0.5
    vx_safe = 0.5
    angle_safe = 0.2
    angvel_safe = 0.2

    vy_excess = max(0.0, abs(vy) - vy_safe)
    vx_excess = max(0.0, abs(vx) - vx_safe)
    angle_excess = max(0.0, abs(angle) - angle_safe)
    angvel_excess = max(0.0, abs(ang_vel) - angvel_safe)

    dynamics_cost = vy_excess + vx_excess + angle_excess + angvel_excess
    soft_landing_penalty = -0.05 * gate * dynamics_cost

    # ---------- Role 3: safe_contact_encouragement ----------
    both_legs_contact = left_contact * right_contact
    x_tol = 0.3
    y_tol = 0.3
    v_tol = 0.2
    angle_tol = 0.1
    angvel_tol = 0.1
    factor_x = max(0.0, 1.0 - abs(x_err) / x_tol)
    factor_y = max(0.0, 1.0 - abs(y_err) / y_tol)
    factor_vx = max(0.0, 1.0 - abs(vx) / v_tol)
    factor_vy = max(0.0, 1.0 - abs(vy) / v_tol)
    factor_angle = max(0.0, 1.0 - abs(angle) / angle_tol)
    factor_angvel = max(0.0, 1.0 - abs(ang_vel) / angvel_tol)
    mean_safety = (factor_x + factor_y + factor_vx + factor_vy + factor_angle + factor_angvel) / 6.0
    safe_contact_bonus = both_legs_contact * mean_safety * 2.0

    total_reward = proximity_reward + soft_landing_penalty + safe_contact_bonus

    components = {
        "proximity_reward": proximity_reward,
        "soft_landing_penalty": soft_landing_penalty,
        "safe_contact_bonus": safe_contact_bonus
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少"接近地面时危险速度/姿态"的前兆 hinge 信号；旧 `soft_landing_penalty` 无差别惩罚所有动态，阻碍探索。
- **behavior**: agent 快速坠毁（全 20 轮 early terminal，len≈68，score≈-110），负分主要由 `soft_landing_penalty` 过度惩罚造成。
- **signal**: `soft_landing_penalty` 对安全动态也施加惩罚（无阈值），且 per-step 惩罚量是主信号的 15–30 倍，严重压制接近目标的正梯度。
- **level**: Level 2 — 将连续惩罚结构变换为 hinge 惩罚，只惩罚超出安全阈值的过剩动态。
- **hypothesis**: hinge 形式在安全区域内给出零惩罚，agent 只需将速度/姿态压入安全包线即可消除所有 penalty，同时保留 `proximity_reward` 的正向拉动，从而在靠近地面时形成"减速接近"的良性梯度组合。
- **risk**: 系数 0.05 可能偏保守（惩罚过弱），导致 agent 在安全边界附近缺乏足够的驱动力进一步减速。若下一轮 len 回升但仍有 crash，可适度提高系数或收紧阈值。
