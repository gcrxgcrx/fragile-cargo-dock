# Response Record

# 设计理由
当前 `proximity_reward` 是一个全时负奖励（`-2.0 * dist/(1+dist)`），**每步都在惩罚存活**。  
这导致 agent 学会 **快速自杀**（len≈68，全部 early terminal）以最小化累计负分，根本原因是“活着就亏钱”。  
本次修改将 `proximity_reward` 从 **绝对值负奖励** 变为 **改善量奖励**：  
`proximity_reward = -5.0 * (f(next_dist) - f(dist))`，  
当 agent 靠近目标时获得正奖励，远离时受到惩罚，原地不动时奖励为 0——**移除每步存活惩罚**，同时保留对接近的梯度。  
系数 5.0 使单步改善奖励量级足以推动学习（预期平均 ~0.5–1.0），而不压过其他组件。  
其他组件 (`soft_landing_penalty`, `safe_contact_bonus`) 保持不变，因为它们的职责（抑制硬着陆、鼓励双足接触）当前未显现破坏性问题。

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
    landing_threshold = 2.0
    gate = max(0.0, 1.0 - abs(y_err) / landing_threshold)
    d_vx = vx**2 / (1.0 + vx**2)
    d_vy = vy**2 / (1.0 + vy**2)
    d_angle = angle**2 / (1.0 + angle**2)
    d_angvel = ang_vel**2 / (1.0 + ang_vel**2)
    dynamics_cost = d_vx + d_vy + d_angle + d_angvel
    soft_landing_penalty = -0.5 * gate * dynamics_cost

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
- **audit**: 缺少对水平越界的显式前兆；但致命的缺口是 `proximity` 全时负奖励诱导快速自杀。
- **behavior**: agent 在 ~68 步终止，全部 early terminal（score < -50），试图尽快结束 episode 以减少累积负分。
- **signal**: 缺失“活着不罚”的接近梯度；原有的绝对距离惩罚迫使 agent 厌恶存活时间。
- **level**: Level 2
- **hypothesis**: 移除每步存活惩罚、改用改善量奖励后，agent 不再有动力提前终止；逐步靠近目标的行动会获得正反馈，结合后续稳定性惩罚与接触奖励，应能恢复学习信号。
- **risk**: 改善系数若偏小可能导致学习缓慢；如果 `safe_contact_bonus` 不足以驱动最终着陆，agent 可能停滞在目标附近而不降落，需后续强化接触奖励。
