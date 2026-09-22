# 1. Search objective
- target_score: 200.000000
- current_score: -74.791329
- gap_to_target: 274.791329
- target_achievement_ratio: -37.396%

# 2. 上一轮奖励函数代码（该轮得分: -74.791329）
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

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Score-Only Feedback Ablation

## Final-policy outcome

score=-74.791329, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-106.901131, -37.473805]

## Evaluation distribution

- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)

# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
智能体需要控制一个 2D 飞行器从视口上方初始位置出发，最终稳定停靠在中央目标平台上。  
**主目标**：安全、准确地降落在目标平台上并稳定停靠（settled）。  
**次目标**：在满足主目标的前提下尽量减少燃料消耗、缩短到达时间。  
**非任务目标**：单纯追求快、单纯节省燃料而不考虑着陆安全，都不可取。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 推测为 float32（通常连续空间默认）
- obs[0]: x_position，水平坐标（相对于目标平台中心），reward_usable: true
- obs[1]: y_position，垂直坐标（相对于平台高度），reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，机体倾斜角，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿接触标志（0/1），reward_usable: true
- obs[7]: right_support_contact，右支撑腿接触标志（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作 0: no_engine，不点火
- 动作 1: left_orientation_engine，点燃一侧姿态发动机，产生偏航力矩
- 动作 2: main_engine，点燃主发动机，向下喷气提供向上推力（抵抗重力）
- 动作 3: right_orientation_engine，点燃另一侧姿态发动机，产生反向偏航力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
  - body_not_awake_or_settled：机体进入休眠或判定为已稳定停靠（可能结合接触和低运动状态），此条件终止可视为潜在成功。
- failure-like termination:
  - crash_or_body_contact：机体主体与地面碰撞或过于猛烈接触导致坠毁。
  - horizontal_position_outside_viewport：水平方向超出视口边界。
- ambiguous termination:
  - body_not_awake_or_settled 在没有足够上下文时也可能是失败（例如倒置卡死），但更常指向成功。
- truncation: 未提及，无。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空字典 {}）
- forbidden_or_uncertain_info_fields: 任何假设的 "success"、"failure"、"landed"、"crash" 标志均不存在于 info 中

## 7. 可用于奖励函数的信号
- position: next_obs[0]（x 误差），next_obs[1]（y 误差）
- velocity: next_obs[2], next_obs[3]
- orientation: next_obs[4]（角度），next_obs[5]（角速度）
- contact: next_obs[6], next_obs[7]
- action/engine: action 取值 0/1/2/3（可推导是否开主发动机、姿态发动机）

# 6. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 7. 历史记忆
# Score-Only Reward Memory

| iter | skeleton | score | best | delta | len |
|---:|---|---:|---:|---:|---:|
| 1 | proximity_reward + safe_contact_bonus + soft_landing_penalty | 5.67 | 5.67 | 0.00 | 84.40 |
| 2 | proximity_reward + safe_contact_bonus + soft_landing_penalty | -119.47 | 5.67 | -125.14 | 68.35 |
| 3 | proximity_reward + safe_contact_bonus + soft_landing_penalty | -111.71 | 5.67 | -117.37 | 68.45 |
| 4 | proximity_reward + safe_contact_bonus + soft_landing_penalty | -9.34 | 5.67 | -15.00 | 1000.00 |
| 5 | distance_shaping + fuel_penalty + orientation_penalty + progress_reward + velocity_constraint | -74.79 | 5.67 | -80.46 | 1000.00 |