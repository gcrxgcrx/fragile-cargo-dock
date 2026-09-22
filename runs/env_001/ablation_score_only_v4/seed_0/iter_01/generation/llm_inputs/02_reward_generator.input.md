# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
该环境是一个 2D 飞行器轨迹优化任务。一个刚体飞行器从视口顶部中央附近开始，带有随机的初始作用力。核心任务是控制飞行器的方向引擎和主引擎，使其飞到视口中央的目标着陆垫上，并尽快、稳定地停靠在垫上。次要目标是完成该过程所用的时间尽可能短，同时使用的发动机推力尽可能少。智能体需要学会逐步接近目标，减速，保持稳定的姿态，并安全接触着陆垫。不应将快速完成或省燃料与原目标（精准停靠）混淆。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务的核心目标是到达并稳定停靠在指定目标位置上，各种操作（姿态调整、推力控制）均服务于该到达-停靠目标。节能和快速属于附属优化要求，而非权重相当的并列目标。

动力学细分子类型 dynamics_subtype: goal_approach_and_soft_contact  
（接近目标并低速、稳定接触/停靠）

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 浮点数（连续值，接触标志为 0.0 或 1.0）
- 各维度含义及奖励可用性：
  - obs[0] (x_position): 飞行器相对于目标垫的水平坐标。reward_usable: true
  - obs[1] (y_position): 飞行器相对于垫基准高度的垂直坐标。reward_usable: true
  - obs[2] (x_velocity): 水平线速度。reward_usable: true
  - obs[3] (y_velocity): 竖直线速度。reward_usable: true
  - obs[4] (body_angle): 机身朝向角度。reward_usable: true（但不建议作为强独立目标）
  - obs[5] (angular_velocity): 角速度。reward_usable: true（可用于姿态稳定性）
  - obs[6] (left_support_contact): 左侧支撑点是否接触的布尔标志（1.0/0.0）。reward_usable: true
  - obs[7] (right_support_contact): 右侧支撑点是否接触的布尔标志（1.0/0.0）。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作详细说明：
  - 动作 0 (no_engine): 不启动任何引擎，依靠惯性滑行。
  - 动作 1 (left_orientation_engine): 点燃一个姿态引擎，用于改变飞行器的方向/角度。
  - 动作 2 (main_engine): 点燃主推进引擎，沿机头方向施加推力，用于移动。
  - 动作 3 (right_orientation_engine): 点燃与动作1相对的姿态引擎，用于反向调整姿态。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` —— 可能意味着飞行器已经静止且可能已停靠，但具体成功条件未知，没有明确的“成功”标签。
- failure-like termination:
  - `crash_or_body_contact` —— 碰撞或非目标位置接触（可能表示严重撞击或侧翻）。
  - `horizontal_position_outside_viewport` —— 水平位置超出视口，即飞出边界。
- ambiguous termination: `body_not_awake_or_settled` 可能为成功（停稳在垫上），也可能为失败（因故障卡住无人为动作），但目前无附加信息佐证。
- truncation: 无。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 中无相关字段）
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不存在，且不得假设存在 `info["success"]`、`info["failure"]`、`info["termination_reason"]` 等。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
- 允许使用：
  - `obs`: 完整的 8 维观测信号
  - `action`: 当前执行的离散动作
  - `next_obs`: 下一步观测（如需要也可参考）
  - `info` 中明确允许的字段（当前无允许字段，不允许使用）
  - `training_progress`：只在环境提示明确允许时使用，此处未提及，禁止使用
- 禁止使用：
  - `original_reward` / 官方奖励
  - 任何未在 observation_space 中声明的 obs 切片
  - 任何未在 env 文档或 source 中显式出现的 info 字段
  - 禁止回推或复现官方奖励

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) —— 直接给出相对目标垫的水平、垂直距离
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) —— 可用于阻尼或安全约束
- orientation: obs[4] (body_angle), obs[5] (angular_velocity) —— 可用于姿态稳定性，但无目标角度
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) —— 着陆接触标志
- action/engine: 当前动作（离散值 0–3），可用于惩罚推力使用
- other: 可组合以上信号构建复合奖励，例如距离、速度、接触的联合条件

## 8. 不确定或不可用的信号
- 明确的成功/失败标志：不存在
- 目标垫的物理边界或相对速度限制：未提供，需基于观察推断
- 发动机推力具体大小、方向：无物理参数，仅知动作效果
- 训练进度/回合数相关的信息：未被允许使用
- 额外的 info 字段（如 `success`, `failure`,



# expert_reward_context.md

# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

不要反过来先套某个 skeleton 名称。模板只提供专家思考方式，不构成封闭候选集合。

---

## 1. Expert Schema 使用规则

- environment_card.md 中的 `expert_task_profile`、`reward_role_decomposition`、`role_to_signal_mapping` 优先级最高。
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

每个算子包含：数学形式、适用场景、触发证据、反模式。

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2` 或 `w * exp_form`
    凸化形式在 signal 较大时提供更强梯度。触发证据：episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小——说明 agent 满足于低水平稳态，需要凸化奖励来打破。
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
    hinge 只在超出安全区间时生效，避免在安全范围内持续惩罚正常波动。触发证据：约束组件的 active_rate≈100% 但 terminated 率仍然很高——说明"全时惩罚"没有给 agent 安全探索空间，它无论怎么调整都被罚。
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：线性正奖励可能导致慢速平台；凸化形式若权重过大可能诱导极端行为；hinge 的 threshold 设太宽则防护不足。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - 平滑压缩: `x / (1 + abs(x))`
  - 倒数衰减: `1 / (1 + k * abs(error))`
  - 线性衰减: `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。
- 触发证据：某个信号的 episode_sum_mean 出现极端值（远大于其他组件），说明无界形式被 exploit。
- 风险：threshold 过小会导致反馈饱和或无梯度。
- 反模式：不要用 bounded_signal 替代 hinge penalty——如果目标是"只在越界时惩罚"，用 dense_state_signal 的 hinge 形式，不要用 bounded 包围。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 触发证据：有明确的目标度量（如到目标的距离）且该度量在 episode 中单调递减时 agent 表现好。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。
- 反模式：不要对速度类信号用 improvement_delta——持续速度本身已经是"进步"，delta 会退化为噪声。

### 2.4 potential_based_shaping
- 适用职责：有明确 potential function 的任务塑形。
- 常见形式：`gamma * Phi(next_obs) - Phi(obs)`
- 使用条件：能够从环境信号定义合理的 Phi。
- 风险：错误 Phi 会误导策略；reward_v1 不默认使用，除非任务天然适合。

### 2.5 quadratic_penalty
- 适用职责：姿态误差、角速度、动作幅度、速度等轻量约束。
- 常见形式：`-w * error**2` 或 `-w * sum(action_i**2)`
- 使用条件：约束信号可观测，且不应压制主学习信号。
- 风险：权重过大会导致 agent_afraid_to_move 或 over_conservative_policy。
- 触发证据：某维度出现高频大幅波动或极端值，但没有触发终止——说明需要轻量抑制而非硬约束。
- 反模式：不要对"有明确安全边界"的信号用 quadratic_penalty（如身体高度必须在 0.2-1.0）。quadratic 从中心开始罚，会让 agent 困在中心不敢动；应改用 hinge 形式只在边界附近生效。

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * gate_factor`，gate_factor 在身体状态恶化时从 1 平滑衰减到 0。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (signal - danger) / margin))`
- 使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
- 触发证据（关键）：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍然 >0——说明 agent 在"先冲后死"，需要 gate 在健康恶化时切断主奖励，而不是加一个独立惩罚。
- 风险：gate 太严格会抑制探索；gate 的衰减区间应设在"接近危险但尚未终止"的范围内。
- 反模式：不要用"加大独立惩罚系数"替代 gate。如果 terminated 是因为身体状态越界，单纯加大该状态的惩罚（Level 1）通常不如将其作为 gate 乘到主奖励上（Level 2），因为惩罚只在越界后才生效，gate 在越界前就开始衰减主信号。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 触发证据：agent 能在各个子条件上分别取得进展，但无法同时满足——说明缺一个"联合满足"的引导信号。
- 风险：乘积容易塌缩（一个 factor 趋近 0 则整体为 0）；使用 `(factor_1 + factor_2 + ...) / n` 或几何平均 `(factor_1 * factor_2 * ...) ** (1/n)` 可缓解。
- 反模式：不要用二值条件做乘积——每个 factor 必须是连续函数，否则乘积退化为稀疏信号。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。

---

## 3. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"直接定位到"该选哪个算子做 Level 2 变换"。
不要求组件名完全匹配；以数学语义和训练表现证据为准。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

