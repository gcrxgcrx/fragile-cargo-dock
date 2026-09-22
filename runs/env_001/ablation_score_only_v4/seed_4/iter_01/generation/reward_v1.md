# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs signals
    x_pos = next_obs[0]   # horizontal position relative to target pad
    y_pos = next_obs[1]   # vertical position relative to pad (0 at pad)
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]   # 1.0 if contact, 0.0 otherwise
    right_contact = next_obs[7]

    # ---- Component A: Goal proximity ----
    # Encourage moving toward center (x=0, y=0)
    dist_sq = x_pos**2 + y_pos**2
    goal_proximity = 2.0 / (1.0 + dist_sq)

    # ---- Component B: Safe landing constraint ----
    # Gate activates when near ground or any leg touches
    # y_pos threshold: below 2.0 the gate gradually rises to 1.0 at y=0
    height_gate = max(0.0, 1.0 - y_pos / 2.0) if y_pos < 2.0 else 0.0
    contact_gate = 0.2 * (left_contact + right_contact)
    landing_gate = height_gate + contact_gate

    # Penalise excessive speed, tilt, and angular velocity when close to ground or in contact
    # Angular velocity is scaled down to be comparable
    motion_cost = x_vel**2 + y_vel**2 + body_angle**2 + 0.1 * ang_vel**2
    safe_landing_penalty = -0.5 * landing_gate * motion_cost

    # ---- Component C: Fuel efficiency ----
    # Small fixed cost for any engine use (actions 1,2,3)
    fuel_penalty = -0.01 if action != 0 else 0.0

    # ---- Total reward ----
    total_reward = goal_proximity + safe_landing_penalty + fuel_penalty

    components = {
        "goal_proximity": goal_proximity,
        "safe_landing_penalty": safe_landing_penalty,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**：`navigation_goal_reaching` / `goal_approach_and_soft_contact`。  
  任务本质是 2D 着陆——将带推力器的刚体精确停靠到目标垫上，要求低速、竖直、双足着地。

- **selected reward roles**（从 `reward_role_decomposition` 中选取）：  
  1. `goal_proximity`（强制）—— 连续引导 agent 靠近目标。  
  2. `safe_landing`（强制）—— 在接近地面时抑制高速、大倾角和角速度。  
  3. `fuel_efficiency_penalty`（条件）—— 极小常驻惩罚，鼓励少用引擎。

- **role_to_signal_mapping**：  
  - `goal_proximity`：使用 `x_position, y_position`，算子为 `bounded_signal` 的倒数衰减形式 `2.0/(1+dist_sq)`。  
  - `safe_landing`：使用 `y_position` 构建高度门控，然后惩罚 `x_vel, y_vel, body_angle, ang_vel`；接触标志也轻度增加门控，保证接触后更平稳。  
  - `fuel_efficiency_penalty`：使用 `action`，算子为 `action_cost`（固定值）。

- **excluded roles 及原因**：  
  - `stability_boost`：其功能已整合进 `safe_landing`，通过接触门控实现接触后更强的角速度/姿态抑制，无须独立组件。  
  - `fast_landing_bonus`：缺少剩余步数或可靠的成功标志，在 v1 中放弃。  
  - `continuous_survival`、`info_based_success`、`contact_only_reward`：环境不适用或信号不可用。

- **为什么没有使用 terminal_success_reward / terminal_failure_penalty**：  
  环境既没有显式 success flag，也没有显式 failure flag，且 `info` 为空，无法可靠构造终端稀疏奖励。

- **哪些职责留到后续迭代**：  
  - 明确的成功近似 `joint_condition_proxy`（多条件组合 soft proxy）可在后续版本引入，以强化“同时满足所有着落条件”的行为。  
  - 动态效率课程（如 `curriculum_weighting`）或更精细的动作平滑代价留到 agent 学会基本着陆后再考虑。

- **训练后应该观察的 failure modes**：  
  1. agent 因 `landing_gate` 过早激活而不敢下降，悬浮在半空或极慢下降。  
  2. 接触到地面后姿态/角速度惩罚不足，导致着陆后弹起或翻倒。  
  3. agent 以高速度硬砸地面再利用 `goal_proximity` 获得短暂高奖励（冲撞作弊）。  
  4. 由于动作代价很小，agent 可能过度使用引擎但整体影响不大；若出现从不使用引擎导致无法控制姿态，可适当调整权重。