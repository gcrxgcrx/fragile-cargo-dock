# ⚠️ 上一版代码验证失败
错误信息：Reward v3 failed validation: 缺少准确函数签名; 没有发现 components/reward_components/reward_terms 字典赋值; warnings: 建议返回 (float(total_reward), components)。当前 wrapper 兼容 float，但 tuple 返回更利于诊断。; 未发现明显的 total_reward/reward 变量名 (record: runs\env_001\ablation_score_only_v4\seed_1\iter_03\generation\validations\reward_v3.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# 设计理由

## 为什么需要修改

1. **审计发现缺口**：
   - 缺少水平越界（failure termination）的前兆信号
   - 缺少直接对准“着陆成功”状态的奖励（contact_proxy 过于稀疏）
   - 缺少动作效率信号（动作空间 n=4 ≥ 3，但无 engine usage penalty）
   - **safe_contact_proxy active_rate < 5%（估计~6.8%），是最直接的僵尸组件证据**

2. **行为证据**：
   - score 14.4 → 目标 200，差距巨大
   - agent 在慢速徘徊（len=982），没有快速失败说明 agent 在主信号 goal_proximity 推动下存活并趋近，但接触/成功信号缺失导致无法完成最终任务
   - 预判：修改 safe_contact_proxy 为连续化，提高 trigger 率，提供朝向成功着陆的梯度

3. **干预选择**：
   - 唯一目标：让 agent 学习“成功停靠”的样子。当前 safe_contact_proxy 太稀疏，每步平均 ≈0.00，active_rate 预估 < 7%，没有提供有效的梯度
   - 选择 Level 2 结构变换：**稀疏二值 proxy → 连续 bounded factor**

4. **为什么还值得继续**：
   - 这是第一轮实际意义上的诊断修改（历史 iter 2 的得分 14.4 来自同一函数的不同系数调优，未做结构变换）
   - 没有连续 3 轮预判 ❌，因此不必重建骨架

## 具体修改

**组件：safe_contact_proxy（现更名 landing_success）**

**旧形态**：
```python
contact_proxy = left_contact * right_contact * exp_neg_speed
```
- 三个因子相乘，任何因子≈0 即整体≈0
- left_contact 和 right_contact 在着陆前几乎总是 0，导致整个项塌缩
- active_rate 极低，只在 perfect landing step 触发一点梯度

**新形态**：
```python
# contact_score: 脚接触率，从 0 到 1 连续
contact_score = (left_contact + right_contact) / 2.0

# near_target: 接近目标的 soft gate
dist = (x**2 + y**2)**0.5
near_target = max(0.0, 1.0 - dist / 1.5)   # 1.5m 内开始激活

# success: 接触+接近 共同作用
landing_success = 2.0 * contact_score * near_target
```
- `contact_score` 连续化（0, 0.5, 1），每步都有梯度
- `near_target` 作为 gate，防止在远处刷脚接触
- gate 选择 hinge 而非指数，因为距离 > 1.5m 时确实不需要学习接触，hinge 提供明确边界
- 增益系数 2.0，使 landing_success 在完美状态下贡献 2.0，与 goal_proximity 量级相当

**为何不移除 exp_neg_speed**：
- velocity_damping 已经全局惩罚速度，不需要在 success 项中重复
- 速度通过 goal_proximity 的成分 vx,vy 已间接推动减速

**为何不加 engine usage penalty 和越界前兆**：
- 一轮只改一个组件，优先修补缺失的核心信号（着陆成功梯度）
- 效率信号（engine penalty）和越界前兆可以在后续迭代添加

## 系数校准

- goal_proximity 在接近目标时 ≈1.0，landing_success 在完美着陆时 max=2.0
- 单组件 ≤ 2x 主信号（2.0 ≤ 2×1.0），满足约束
- velocity_damping 当前系数 -0.2，per-step ≈ -0.40（根据 episode_sum_mean/len 推断）。需降低到 ≤0.3x 主信号（0.3×0.25=0.075）。将 velocity_damping 系数从 -0.2 降至 -0.05。

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
    total = goal_proximity + landing_success + velocity_damping + angle_penalty + angvel_penalty

    components = {
        'goal_proximity': goal_proximity,
        'landing_success': landing_success,
        'velocity_damping': velocity_damping,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty
    }

    return float(total), components
```

# 诊断摘要
- **audit**: safe_contact_proxy 是僵尸组件（active_rate≈6.8%），缺向着陆成功的梯度；缺越界前兆和效率信号（下次处理）
- **behavior**: agent 慢速徘徊（len=982），在 goal_proximity 推动下存活趋近，但缺乏成功停靠信号导致无法完成最终目标
- **signal**: landing success 信号缺失（旧 contact_proxy 太稀疏，无有效梯度）
- **level**: Level 2 — 结构变换（稀疏二值 proxy → 连续 bounded factor，增加 gate）
- **hypothesis**: 连续 contact_score + near_target gate 将提供每步都有的着陆梯度，引导 agent 从“接近目标”过渡到“双脚着地”，score 应提升至 30-50 区间
- **risk**: agent 可能在 1.5m 以内把脚放下来但不完成着陆（悬挂接近），需要后续加入速度 gate 或 stay-alive 截止项来推动最终完成；velocity_damping 降系数后速度可能上升，但当前牺牲一定速度控制换取成功停靠梯度是合理的取舍

# 1. Search objective
- target_score: 200.000000
- current_score: 14.396573
- gap_to_target: 185.603427
- target_achievement_ratio: 7.198%

# 2. 上一轮奖励函数代码（该轮得分: 14.396573）
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

    # Exponential helpers (no imports allowed)
    exp_neg_speed = 2.718281828 ** (-speed)

    # 1. Decomposed proximity – vertical descent reward * horizontal alignment
    vertical_proximity = 2.718281828 ** (-abs(y))          # [0,1], peaks when y→0
    horizontal_alignment = 2.718281828 ** (- (x**2))      # [0,1], peaks when x→0, narrow sigma
    goal_proximity = vertical_proximity * horizontal_alignment  # joint incentive

    # 2. Safe contact proxy – joint condition: both contacts and low speed
    contact_proxy = left_contact * right_contact * exp_neg_speed

    # 3. Velocity damping – distance‑gated speed penalty (kept for safety)
    velocity_damping = -0.2 * speed * (2.718281828 ** (- (x**2 + y**2)**0.5))

    # 4. Orientation stability – reduced penalties to avoid dominating reward
    angle_penalty = -0.1 * (angle ** 2)      # was -0.5, now -0.1
    angvel_penalty = -0.1 * (angvel ** 2)    # kept at -0.1

    # Aggregate reward
    total = goal_proximity + contact_proxy + velocity_damping + angle_penalty + angvel_penalty

    components = {
        'goal_proximity': goal_proximity,
        'safe_contact_proxy': contact_proxy,
        'velocity_damping': velocity_damping,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty
    }

    return float(total), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. 训练反馈
# Score-Only Feedback Ablation

## Final-policy outcome

score=14.396573, len=982.150000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-34.297140, 170.295302]

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