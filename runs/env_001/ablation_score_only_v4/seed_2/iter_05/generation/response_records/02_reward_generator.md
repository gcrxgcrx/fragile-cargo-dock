# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------- unpack observations --------------------
    # obs layout: [x, y, vx, vy, angle, av, left_contact, right_contact]
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # -------------------- distance to target pad --------------------
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # -------------------- weights --------------------
    w_progress = 2.0
    w_vel = 2.0
    w_angvel = 0.5
    w_angle = 0.5
    w_contact = 0.5
    k_proximity = 3.0   # controls how fast velocity penalty ramps up near the pad

    # -------------------- 1. progress reward (improvement_delta) --------------------
    progress_reward = w_progress * (dist - next_dist)

    # -------------------- 2. velocity penalty (soft, proximity‑gated) --------------------
    proximity_gate = 1.0 / (1.0 + k_proximity * next_dist)
    speed_sq = nvx**2 + nvy**2
    vel_penalty = -w_vel * speed_sq * proximity_gate
    ang_penalty = -w_angvel * (n_angvel**2) * proximity_gate

    # -------------------- 3. upright orientation incentive --------------------
    orientation_penalty = -w_angle * (n_angle**2)

    # -------------------- 4. contact reward --------------------
    contact_reward = w_contact * (n_left_contact + n_right_contact)

    # -------------------- total reward --------------------
    total_reward = (progress_reward +
                    vel_penalty +
                    ang_penalty +
                    orientation_penalty +
                    contact_reward)

    components = {
        "progress": progress_reward,
        "velocity_penalty": (vel_penalty + ang_penalty),
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务与动力学子类型
- **selected task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact` – 飞行器/着陆器需要在接近目标平台时减速、对准并实现软接触。

## 2. 选用的奖励角色 (reward roles)
环境卡片将以下角色标记为 **mandatory**，本条目的 v1 版本全部实现：

1. **goal_distance_encouragement** → `progress` 主学习信号  
2. **soft_landing_velocity_penalty** → `velocity_penalty` (带 proximity‑gate)  
3. **upright_orientation_incentive** → `orientation_penalty`  
4. **contact_reward** → `contact_reward`

未引入效率/燃料代价（secondary objective），留在后续迭代。

## 3. 职责–信号映射

| 角色 | 使用的观测信号 | 公式算子 | 设计理由 |
|------|----------------|----------|----------|
| `progress` | `(x, y)` → 当前/下一步距离 | `improvement_delta` (`old – new`) | 稠密、每步梯度，直接奖励向平台靠近；不使用静态 `-distance` 以避免“悬停也可以得分”的捷径。 |
| `velocity_penalty` | `next_vx, next_vy, next_angvel` + `next_dist` | `dense_state_signal` (二次惩罚) × **proximity‑gate** | 采用 `-w * (vx²+vy²+av²) * 1/(1+k*dist)`，使速度惩罚仅在靠近平台时生效，远距离时不压制必要的下降和水平移动探索。 |
| `orientation_penalty` | `next_body_angle` | `dense_state_signal` (二次惩罚) | 轻量二次罚将机体约束在竖直附近，防止倾斜过大导致侧翻或无效主推力。 |
| `contact_reward` | `next_left_contact, next_right_contact` | 离散正奖励 | 二值接触标志直接作为稀疏完成信号；结合速度惩罚阻止 agent 采用硬着陆刷分。 |

## 4. 排除的角色及原因
- **terminal_success_reward**：`explicit_success_flag_available = false`，`info` 为空，无法在终端步给出确信的成功加分。若强行构造可能将不稳定的“settled”误判为成功，引入噪声。
- **terminal_failure_penalty**：同样缺少显式失败标志。
- **fuel_cost / action magnitude penalty**：次要目标，当前版本优先让 agent 学会安全着陆；过早引入燃料惩罚会导致 agent 不敢使用引擎（已在历史尝试中证实）。
- **soft_health_gate（乘到主奖励上）**：主进度信号 `progress` 本身会因速度惩罚而受到制约；独立的健康 gate 会增加设计的耦合度，v1 保持组件独立、可消融。

## 5. 未包含但在后续迭代中可考虑的职责
- 燃料/动作代价（`action_penalty`）→ 在 agent 可靠着陆后再引入。
- 更精巧的 **joint‑condition proxy**（如 `proximity × speed × orientation × contact` 乘积）→ 当 agent 能分别满足子条件但无法同时完成时再加入。
- 动态课程权重（`curriculum_weighting`）→ 需要稳定 training progress 参考，v1 不启用。

## 6. 训练后需观察的 failure modes
- **hovering / slow descent**：若 `progress` 权重过低，agent 可能因 velocity penalty 不敢加速度，导致 episode 过长。应检查 `progress` 绝对值是否持续很小。
- **late‑stage oscillations**：接近平台时速度惩罚 + 距离 delta 可能引起反复加减速。可观察 velocity 的频谱或最终 landing 的震荡次数。
- **contact reward dominance**：若接触奖励过大，agent 可能忽略中间减速，直接高速冲下（但因为同时有速度惩罚，这一模式会被抑制；需验证惩罚权重是否足够）。
- **under‑use of main engine**：进度过于依赖重力，agent 可能极少点燃主引擎，只在最后接触时靠无引擎下落——此时纵向速度可能偏大而触发 velocity penalty。若出现大量 `main_engine` 未被使用的回合，说明进度项需要更强引导。

## 7. 设计假设与 risky decisions
- **不使用 `terminal_success_reward`**：我们假设良好的 step 级稠密奖励足以引导终止行为；若训练后期难以完成 episode，可能需要引入 soft proxy 辅助。
- **velocity penalty 靠近似门控而非全时惩罚**：避免了早期压制探索，但门控参数 `k_proximity` 需要微调；若取值过大（gate 陡峭），transition 可能出现梯度陡变。
- **`progress` 允许负值**：当 agent 反向移动时 `dist - next_dist < 0` 会直接变为惩罚，这在简单任务中合理；若任务要求大量水平绕行，可考虑 `max(0, diff)`，但当前平台位于正下方，基本为单调接近。
