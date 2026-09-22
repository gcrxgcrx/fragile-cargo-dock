# Response Record

# 设计理由
从累积迭代记录看，iter5 的 `contact_reward`（直接奖励支撑腿着地）取得了 150.41 的高分，而 iter6/7 将其替换为带速度‑角度折减因子的乘积形式 `landing_reward` 后分数暴跌至 ‑13 ~ 13，且 episode 几乎全部被截断（terminated 仅 3/20）。这说明当前 `landing_reward` 的数学形态——`contact_sum * 1/(1+5·speed) * 1/(1+5·|angle|)`——在训练初期极易塌缩为极小的数值，无法提供足够的学习梯度，导致 agent 永远学不会可靠着陆，只能在高空徘徊。

本轮修改只改 **landing_reward** 组件，把“乘积 proxy” 改为 **加性 hinge 组合**，使 agent 在腿部接触时无论如何都能获得基础正向信号，而速度、姿态只作为额外奖励，不再让奖励坍缩为零。这样做是为了恢复类似 iter5 的强着陆激励，同时保留对低速竖直着陆的偏好。

**数学形式**
- `contact_sum` 仍为 0/1/2。
- `speed_bonus = max(0, 1 - k_speed * speed)`，`k_speed=2.0`（比之前的 5.0 宽，使不够完美的速度也能获得部分额外奖赏）。
- `angle_bonus = max(0, 1 - k_angle * abs_angle)`，`k_angle=2.0`。
- 总因子 = `1.0 + speed_bonus + angle_bonus`，因此基础值至少为 1.0，最大可达 3.0。
- 权重 `w_landing = 2.0`，使得单次完美着陆奖励约为 `2.0 * 2 * 3.0 = 12.0`，与多步进度奖励相比不会过分主导，但足以提供清晰的着陆导向。

**系数校准**
- `k_speed=2.0`：当 speed=0.5 时 bonus=0，阈值设在合理范围内。
- `k_angle=2.0`：当角度偏差 0.5 rad 时 bonus=0。
- 总额外奖励幅度 ≤ 1 倍进度奖励的累积值，符合设计约束。

（注意：代码中保持原有的 `progress_reward`、`velocity_penalty`、`orientation_penalty` 不变。）

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------- unpack observations --------------------
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

    # -------------------- derived quantities --------------------
    speed = (nvx**2 + nvy**2) ** 0.5
    abs_angle = abs(n_angle)

    # -------------------- weights --------------------
    w_progress = 2.0
    w_vel = 0.2
    w_angvel = 0.1
    w_angle = 0.5
    w_landing = 2.0          # increased to give meaningful landing signal
    k_speed = 2.0            # relaxed from 5.0
    k_angle = 2.0            # relaxed from 5.0

    # -------------------- 1. progress reward (unchanged) --------------------
    progress_reward = w_progress * (dist - next_dist)

    # -------------------- 2. velocity penalty (local gate, unchanged) --------------------
    local_gate = max(0.0, 1.0 - next_dist)
    speed_sq = nvx**2 + nvy**2
    vel_penalty = -w_vel * speed_sq * local_gate
    ang_penalty = -w_angvel * (n_angvel**2) * local_gate

    # -------------------- 3. upright orientation incentive (unchanged) --------------------
    orientation_penalty = -w_angle * (abs_angle**2)

    # -------------------- 4. safe landing reward (modified: additive hinge) --------------------
    contact_sum = n_left_contact + n_right_contact
    # Hinge bonuses: 1 when perfect, 0 when outside threshold
    speed_bonus = max(0.0, 1.0 - k_speed * speed)
    angle_bonus = max(0.0, 1.0 - k_angle * abs_angle)
    # Additive combination guarantees base reward >= contact_sum
    landing_factor = 1.0 + speed_bonus + angle_bonus
    landing_reward = w_landing * contact_sum * landing_factor

    # -------------------- total reward --------------------
    total_reward = (progress_reward +
                    vel_penalty +
                    ang_penalty +
                    orientation_penalty +
                    landing_reward)

    components = {
        "progress": progress_reward,
        "velocity_penalty": (vel_penalty + ang_penalty),
        "orientation_penalty": orientation_penalty,
        "landing_reward": landing_reward
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 着陆信号为乘积形式，极易塌缩为 0，且折扣因子过大导致成功着陆几乎没有奖励梯度。
- **behavior**: 在目标附近徘徊但无法完成着陆（971 步截断，terminated 仅 3/20）。
- **signal**: 缺少强健的着陆驱动信号，当前 landing_reward 对不完美的速度和角度惩罚过重以致消失。
- **level**: Level 2
- **hypothesis**: 将乘积改为加性 hinge 后，接触瞬间总能获得基础奖励，使 agent 有动力去尝试着陆；同时额外奖励保留了对低速、竖直状态的偏好，应能显著提高成功率并推高分数。
- **risk**: 如果基础奖励过强，agent 可能过早触地而 crash，但 orientation_penalty 和 velocity_penalty 会抑制这种行为。若见效，后续可微调 k_speed/k_angle 以平衡稳与快。
