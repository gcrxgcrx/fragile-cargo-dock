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
- current_score: 202.180509
- gap_to_target: -2.180509
- target_achievement_ratio: 101.090%

# 2. 上一轮奖励函数代码（该轮得分: 202.180509）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next observation
    x = next_obs[0]           # horizontal position (relative to target)
    y = next_obs[1]           # vertical position (relative to pad)
    vx = next_obs[2]          # horizontal velocity
    vy = next_obs[3]          # vertical velocity
    angle = next_obs[4]       # body angle
    angvel = next_obs[5]      # angular velocity
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Derived quantities
    speed = (vx**2 + vy**2)**0.5
    dist = (x**2 + y**2)**0.5

    # 1. Decomposed proximity
    vertical_proximity = 2.718281828 ** (-abs(y))          # [0,1]
    horizontal_alignment = 2.718281828 ** (- (x**2))       # [0,1]
    goal_proximity = vertical_proximity * horizontal_alignment

    # 2. Landing success – continuous contact score * near-target gate
    contact_score = (left_contact + right_contact) / 2.0   # [0, 0.5, 1]
    near_target = max(0.0, 1.0 - dist / 1.5)               # [0,1], hinge at 1.5m
    landing_success = 2.0 * contact_score * near_target

    # 3. Velocity damping – reduced coefficient to lower penalty burden
    velocity_damping = -0.05 * speed * (2.718281828 ** (-dist))

    # 4. Orientation stability
    angle_penalty = -0.1 * (angle ** 2)
    angvel_penalty = -0.1 * (angvel ** 2)

    # Aggregate reward
    total_reward = goal_proximity + landing_success + velocity_damping + angle_penalty + angvel_penalty

    components = {
        'goal_proximity': goal_proximity,
        'landing_success': landing_success,
        'velocity_damping': velocity_damping,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Score-Only Feedback Ablation

## Final-policy outcome

score=202.180509, len=468.800000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-53.435261, 255.319283]

## Evaluation distribution

- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)

# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主任务：控制一个2D刚体（搭载推力器）从视口顶部中央附近出发，**到达并稳定、安全地停靠在画面中央的目标垫上**。  
次任务：在满足主任务的前提下，**尽可能少地使用引擎推力**（省燃料），并**尽快完成**。  
不应混淆的目标：纯粹的燃料最小化或最短时间不应牺牲安全接触与姿态稳定，着陆必须平稳。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 （推断，标准连续空间）
- obs[0]: `x_position`，水平相对坐标（可能是到目标垫中心的横向距离），可用于奖励（希望→0）
- obs[1]: `y_position`，垂直相对坐标（相对于垫的高度），可用于奖励（希望→0）
- obs[2]: `x_velocity`，水平线速度，可用于奖励（希望→0）
- obs[3]: `y_velocity`，竖直线速度，可用于奖励（着陆时希望→0或很小负值，但通常希望为0）
- obs[4]: `body_angle`，刚体朝向角度，可用于奖励（希望→0，保持直立）
- obs[5]: `angular_velocity`，角速度，可用于奖励（希望→0）
- obs[6]: `left_support_contact`，左侧支撑接触标志，0或1，可用于奖励（着陆成功时期望=1）
- obs[7]: `right_support_contact`，右侧支撑接触标志，0或1，可用于奖励（着陆成功时期望=1）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine`（无推力，依靠惯性滑行）
- action 1: `left_orientation_engine`（向左定向推力，可能影响角速度或横向平移）
- action 2: `main_engine`（主推力，通常向上喷气，产生主要向上加速度，对抗重力）
- action 3: `right_orientation_engine`（向右定向推力，与左引擎对称）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled`  
  含义：刚体不再“醒着”（动能很低、角度稳定、接触垫子后进入休眠或标记为已停靠）。推测当成功着陆并稳定后触发。
- **failure-like termination**:
  - `crash_or_body_contact`：身体发生异常碰撞（可能并非目标垫，例如撞到地面、墙壁或其他部分）
  - `horizontal_position_outside_viewport`：水平位置超出允许范围，飞出视野
- **ambiguous termination**: 无
- **truncation**: 未给出 max steps 信息，可能不存在时间截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** （info 为空，无 `success` 字段）
- explicit_failure_flag_available: **false** （无 `failure` 字段）
- allowed_info_fields: **无** （`info = {}`，禁止使用任何额外信息）
- forbidden_or_uncertain_info_fields: 任何未在 source 中声明为可用的字段，包括 `success`, `failure`, `terminal_observation` 等

## 7. 可用于奖励函数的信号
- **position**: `obs[0]` (x), `obs[1]` (y) — 可计算到目标垫的距离，鼓励趋近于0
- **velocity**: `obs[2]` (vx), `obs[3]` (vy) — 可惩罚绝对值以减慢速度，特别是接近目标时
- **orientation**: `obs[4]` (angle) — 可惩罚偏离直立的角度
- **angular velocity**: `obs[5]` — 可惩罚快速旋转
- **contact**: `obs[6]` (left), `obs[7]` (right) — 用于设计着陆成功奖励或接触条件，期望双侧均为1且速度很小
- **action / engine usage**: `action` 本身 — 可惩罚使用主引擎或所有引擎的情形，促进燃料效率
- **other**: 可从位置、速度、接触组合出复合信号（如“已着陆标志”：两个接触且速度/角度在阈值内）

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
| 1 | angle_penalty + angvel_penalty + goal_proximity + safe_contact_proxy + velocity_damping | 11.01 | 11.01 | 0.00 | 780.50 |
| 2 | angle_penalty + angvel_penalty + goal_proximity + safe_contact_proxy + velocity_damping | 14.40 | 14.40 | 0.00 | 982.15 |
| 3 | angle_penalty + angvel_penalty + goal_proximity + landing_success + velocity_damping | 202.18 | 202.18 | 0.00 | 468.80 |
```
