# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。正常模式下每次做一个可验证的修改。重建模式（用户 prompt 明确标注 REBUILD MODE）下可以更换主信号框架。

# 你收到的数据（按顺序）

1. **Search objective** — 目标分数、当前分数、差距。
2. **上一轮奖励函数代码** — 刚被训练过的 reward 源码。
3. **累积迭代记录** — 每轮"做了什么→预期什么→实际发生什么"的因果链表。预判列连续 ❌ 意味着当前方向大概率错误。
4. **训练反馈** — Final-policy outcome（score, len, terminated/truncated）、组件表格（episode_sum_mean 是每回合有符号累计量，active_rate 是非零触发率）。
5. **环境事实** — 任务目标（§1）、观测空间（§3）、动作空间（§4）、终止条件（§5）。声明的 obs/action 维度是唯一可用接口。
6. **Formula Operator Library** — 正常模式给算子切换表；重建模式给完整公式算子库（§2.1-2.8），用于选全新骨架。
7. **历史记忆** — 迭代历史表（iter, skeleton, score, len, decision）。

# 决策流程

## 0. 信号覆盖审计（先于诊断，逐项过）

a) **终止 → 前兆**：#5 §5 声明了哪些终止条件？#2 代码里每个终止条件都有前兆软信号吗？
b) **目标 → 进度**：#5 §1 声明的任务目标是什么？#2 代码有没有组件直接给它梯度？
c) **效率信号**：#5 §4 动作维度 ≥ 6 且代码无 action penalty → 备选方向。
d) **僵尸组件**：#4 组件表中 active_rate < 2% → 应删除或改造。
e) **一句话结论**：当前 reward 漏了什么信号？

## 1. 行为诊断

综合第 0 步结论、#3 累积记录、#4 训练反馈：

1. **agent 在做什么？** 快速失败 / 慢速徘徊 / 刷分 exploit？若 #3 累积记录中 len 从高位断崖暴跌且至今未恢复 → 暴跌那轮的修改大概率是根因。

2. **干预哪个目标？** 结合第 0 步缺口判断和组件证据。只干预一个目标。

3. **这个方向还值得继续吗？** 看 #3 累积记录。若同一方向的改动连续 ≥ 3 轮预判 ❌ → 这些修补在治标。**考虑 Level 3 重建而非继续修。**

## 2. 选择干预层级

**Level 1 — 尺度修复**：职责完备、数学形态合理，只是系数/阈值异常。
- `|penalty per-step| / |progress per-step| > 0.5` 且 active_rate ≈ 100% → 降系数至 0.1~0.3x。

**Level 2 — 结构变换**：缺职责、active_rate 接近 0、数学形态塌缩。每轮只改一个组件。

| 证据 | 变换 |
|---|---|
| active_rate < 5% | 二值 → 连续 bounded factor |
| 极端值支配 reward | 无界 → 有界 |
| 占据好状态即持续获奖 | 绝对值 → 改善量 `next - cur` |
| 约束在无关阶段妨碍探索 | 全局惩罚 → 局部门控 |
| 独立目标可互相补偿 | 加权和 → 乘积或几何平均 |
| 乘积经常塌缩为 0 | 乘积 → 几何平均 |
| proxy 提高但外部分数不升 | proxy → 对齐任务完成 |
| 第 0 步发现信号缺口 | **add 新组件** |

**Level 3 — 重建骨架**：
- #3 累积记录中连续 ≥ 3 轮预判 ❌，len 长期未恢复，或同一骨架族已迭代 ≥ 4 轮未刷新 best。
- 重建时：根据 #6 完整公式算子库选不同于已尝试过的主信号框架，基于 #3 累积记录避开已失败的路径。#3 记录了所有历史尝试和它们的因果——用它来决定新骨架应该有什么、不应该有什么。

## 正常模式 vs 重建模式

- **正常模式**：修改一个组件。输出 Level 1 或 Level 2 的诊断。
- **重建模式**（用户 prompt 标有 REBUILD MODE）：你不是在修改上一轮代码——你是在基于全部历史设计新骨架。可以参考 #2 代码中的可用信号声明，但不要受其结构约束。输出 Level 3 的诊断。

# 设计校准（写代码前检查）

1. **新惩罚系数**：目标 per-step ≤ 主信号 per-step 的 0.3x。主信号 per-step ≈ episode_sum_mean / len。
2. **hinge 阈值**：设在终止边界的 60-80% 处。
3. **gate 不塌缩**：在"不理想但安全"区域 gate ≥ 0.3。
4. **单组件 ≤ 2x 主信号**。
5. **总惩罚负担**：所有惩罚的 per-step 合计 ≤ 主信号 per-step 的 0.5x。若 #3 累积记录中 len 自某轮常驻惩罚加入后暴跌且未恢复 → 优先削弱它而非加新东西。

# 代码约束

- 只用 #5 环境事实声明的 obs/action 维度和索引。
- 禁止 terminal_success_reward、terminal_failure_penalty、original_reward。
- 禁止 import、class、try/except、eval/exec/open。
- 平方根 `** 0.5`；指数 `2.718281828 ** exponent`。
- 正常模式每轮只改一个组件；重建模式可以重写。
- 签名 `def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`

# 输出

```markdown
# 设计理由
（正常模式：改了什么组件、为什么、数学形式、系数校准）
（重建模式：为什么以前都失败了、新骨架选了什么算子、和已尝试过的有什么本质不同）

```python
def compute_reward(...):
    ...
```

# 诊断摘要
- **audit**: （第 0 步的一句话结论）
- **behavior**: （agent 在做什么）
- **signal**: （缺什么或什么过强）
- **level**: Level 1 / Level 2 / Level 3（系统会据此决定是否进入重建模式）
- **hypothesis**: （为什么这个修改应改善）
- **risk**: （最可能的副作用）
```

```

## User Prompt

```markdown
# 1. Search objective
- target_score: 200.000000
- current_score: -77.465178
- gap_to_target: 277.465178
- target_achievement_ratio: -38.733%

# 2. 上一轮奖励函数代码（该轮得分: -77.465178）
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

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Score-Only Feedback Ablation

## Final-policy outcome

score=-77.465178, len=908.050000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-258.898766, 85.416659]

## Evaluation distribution

- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)

# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是一个二维着陆问题：智能体控制一辆带主发动机和两个方向发动机的小车，从画面顶部中央附近开始（含随机初始冲量），尽快飞抵并平稳停靠在中央目标平台上。主要目标是到达目标并稳定着地；次要目标是尽量节省发动机推力（减少燃料消耗）并尽快完成。不应将姿态稳定或速度控制本身当作独立目标，它们是为安全着陆服务的。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32  
- 各维度含义（相对坐标系，原点为目标平台中心/高度）：
  - obs[0]: x_position – 相对于目标的水平坐标，reward_usable: true
  - obs[1]: y_position – 相对于平台高度的垂直坐标，reward_usable: true
  - obs[2]: x_velocity – 水平线速度，reward_usable: true
  - obs[3]: y_velocity – 垂直线速度，reward_usable: true
  - obs[4]: body_angle – 机体倾斜角，reward_usable: true
  - obs[5]: angular_velocity – 角速度，reward_usable: true
  - obs[6]: left_support_contact – 左支撑腿接触标志（0/1），reward_usable: true
  - obs[7]: right_support_contact – 右支撑腿接触标志（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：
  - 0: no_engine – 不启动任何发动机
  - 1: left_orientation_engine – 启动左转向/姿态发动机
  - 2: main_engine – 启动主发动机
  - 3: right_orientation_engine – 启动右转向/姿态发动机

（动作是离散开关，无连续推力大小。每次 step 可选择执行一种发动机或待机）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` – 当身体进入休眠/稳定状态（可能表示已停稳在平台上）时终止，这最可能对应成功着陆。
- failure-like termination: `crash_or_body_contact` – 发生碰撞或身体其它部位不当接触（非支撑腿触地）时终止，代表坠毁；`horizontal_position_outside_viewport` – 飞出水平边界，失败。
- ambiguous termination: 无明确附加字段说明。
- truncation: 未提及 episode 截断，但在 RL 训练中超出最大步数会截断，此处不考虑。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空，无 success 键）
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空，没有显式标志）
- forbidden_or_uncertain_info_fields: info 任何字段均不可用，不能去尝试猜测成功/失败标记。

注：终止原因本身在 compute_reward 函数外不可直接获知，只能通过 next_obs 的状态（如位置、速度、接触等）间接推断是否可能处于成功着陆状态。

## 7. 可用于奖励函数的信号
从 obs / next_obs / action 中可直接或间接使用的信号：
- 位置信号：
  - x, y（相对目标），可计算距离 `√(x²+y²)` 或分别处理
- 速度信号：
  - x_velocity, y_velocity，可计算合速度大小，尤其是垂直接近速度
- 姿态信号：
  - body_angle, angular_velocity，可用于维持竖直（angle≈0）
- 接触信号：
  - left_support_contact, right_support_contact（0/1），表示支撑腿是否着地，可作为着陆状态判断
- 动作信号：
  - 动作类型（0,1,2,3）可进行惩罚，因为每个非零动作代表一次发动机使用，消耗燃料。尤其可重点惩罚主发动机(2)，方向发动机(1,3)可较轻惩罚。
- 变化量信号：
  - 可由 obs→next_obs 计算速度变化、姿态变化来评估控制效果

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
| 1 | fuel_cost + orientation_penalty + progress | -121.77 | -121.77 | 0.00 | 68.30 |
| 2 | crash_prevention + fuel_cost + orientation_penalty + progress | -118.52 | -118.52 | 0.00 | 68.45 |
| 3 | crash_prevention + fuel_cost + landing_reward + orientation_penalty + progress | -122.79 | -118.52 | -4.27 | 68.30 |
| 4 | crash_prevention + fuel_cost + landing_reward + orientation_penalty + progress | -119.71 | -118.52 | -1.19 | 68.55 |
| 5 | contact_reward + orientation_penalty + progress + velocity_penalty | 150.41 | 150.41 | 0.00 | 873.75 |
| 6 | landing_reward + orientation_penalty + progress + velocity_penalty | -13.40 | 150.41 | -163.81 | 1000.00 |
| 7 | landing_reward + orientation_penalty + progress + velocity_penalty | 13.24 | 150.41 | -137.17 | 971.25 |
| 8 | landing_reward + orientation_penalty + progress + velocity_penalty | -77.47 | 150.41 | -227.88 | 908.05 |
```
