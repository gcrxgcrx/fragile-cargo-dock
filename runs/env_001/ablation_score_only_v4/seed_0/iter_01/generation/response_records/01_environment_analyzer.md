# Response Record

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
