# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 计算到目标的距离 ---
    dist_old = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_new = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    delta_dist = dist_old - dist_new                      # 正值表示靠近目标

    progress_reward = 1.0 * delta_dist                    # 主学习信号

    # --- 速度软约束（超过阈值才惩罚）---
    speed_new = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    speed_threshold = 0.5
    vel_excess = max(0.0, speed_new - speed_threshold)
    velocity_penalty = -0.5 * (vel_excess ** 2)           # 二次惩罚，仅作用于超速部分

    # --- 角速度轻量惩罚 ---
    ang_vel = next_obs[5]
    angular_penalty = -0.1 * (ang_vel ** 2)               # 抑制剧烈旋转

    # --- 软着陆近似：靠近 + 低速 + 姿态稳定 ---
    dist_thresh = 0.3
    speed_proxy_thresh = 0.3
    ang_thresh = 0.2
    f_dist  = max(0.0, 1.0 - dist_new / dist_thresh)
    f_speed = max(0.0, 1.0 - speed_new / speed_proxy_thresh)
    f_ang   = max(0.0, 1.0 - abs(ang_vel) / ang_thresh)
    landing_proxy = (f_dist + f_speed + f_ang) / 3.0     # 平均避免乘积塌缩

    total_reward = progress_reward + velocity_penalty + angular_penalty + landing_proxy

    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "angular_penalty": angular_penalty,
        "landing_proxy": landing_proxy
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **task_family / dynamics_subtype**: `navigation_goal_reaching` / `goal_approach_and_soft_contact`。核心是让飞行器逐步靠近着陆垫并低速稳定停靠。
- **selected reward roles**:
  1. **主学习信号 (mandatory)**: `approach_progress`，使用 `improvement_delta` 算子，奖励每一帧缩短的距离。
  2. **安全/稳定约束 (conditional)**: 
     - `velocity_penalty`：采用 `dense_state_signal (hinge quadratic)`，仅惩罚超过阈值（0.5）的速度，避免安全探索被压制。
     - `angular_penalty`：采用 `quadratic_penalty`，轻量抑制角速度抖动。
  3. **任务完成近似 (conditional)**: `landing_proxy`，使用 `joint_condition_proxy`（三个连续因子的均值），当飞行器同时满足靠近目标、低速、低角速度时给予额外正反馈，引导停靠行为。
- **role_to_signal_mapping**:
  - progress: `obs[0,1]` 与 `next_obs[0,1]` → 距离差值。
  - velocity_penalty: `next_obs[2,3]` → 合成速度大小。
  - angular_penalty: `next_obs[5]` → 角速度。
  - landing_proxy: 综合 `next_obs[0,1]`（距离）、`next_obs[2,3]`（速度）、`next_obs[5]`（角速度）。
- **excluded roles 及原因**:
  - `terminal_success_reward` / `terminal_failure_penalty`: 环境无显式成功/失败标志，`info` 为空。
  - `action_cost` (效率/动作代价): 遵循 v1 原则，留到后续迭代；避免因省燃料过度抑制机动。
  - `soft_health_gate`: 当前终止主要因碰撞或飞出视口，但未提供明确的“健康程度”信号，不宜在 v1 引入。
- **未使用 terminal 信号说明**: 由于缺乏 `explicit_success_flag` 和 `explicit_failure_flag`，无法可靠区分成功与失败终止，故不设置终端奖励/惩罚。
- **留到后续迭代的职责**:
  - 推力效率/发动机代价（动作2成本较高，动作1/3 成本较低）。
  - 更精细的接触条件（例如要求两个支撑点同时接触才触发额外奖励）。
  - 动态课程权重（如果早期冲撞严重可后期增加姿态/速度要求）。
- **训练后应观察的 failure modes**:
  - agent 可能学会在视口内慢速徘徊（速度低于阈值，距离不缩小），需检查 `progress_reward` 是否仍能推动接近。
  - 可能出现过早“着陆”但实际仍在半空悬停（接触标志未真触发），此时 `landing_proxy` 可能给分但任务未完成，需监控实际接触率。
  - 若角速度惩罚过重，可能导致姿态调整不足，主引擎推力方向错误，建议保持较小系数。