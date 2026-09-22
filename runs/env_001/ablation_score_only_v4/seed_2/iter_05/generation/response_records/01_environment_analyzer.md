# Response Record

# 匿名环境理解卡片

## 1. 任务目标
该匿名环境为 2D 飞行器轨迹优化任务，主体受到重力作用，从视口顶部中心附近开始，并带有随机初始作用力。核心目标是**尽快到达并稳定停靠在画面中央的目标平台上**，要求接近平台时减速、保持竖直姿态并实现安全软接触。次要目标是**尽量节约引擎推力**（即减少执行器使用），但不得因此牺牲着陆成功率或造成危险。

## 2. 任务类型选择
- selected_route_id: `navigation_goal_reaching`
- confidence: `high`
- reason: 任务的核心驱动力是到达一个明确的目标位置（中心平台），并稳定下来。着陆姿态、接触条件、燃料效率均为服务于该主要目标的附属要求；不存在多个权重相当的冲突性核心目标，因此归为导航/目标到达族。

动力学子类型进一步确定为 `goal_approach_and_soft_contact`，因为航天器/飞行器需要在接近目标时减速、旋转对准并实现低速、稳定的接触。

## 3. 观察空间 observation_space
- type: `Box`
- shape: `[8]`
- dtype: `float32`（默认连续环境）
- 各维度含义：

| index | 名称 | 含义 | reward_usable |
|-------|------|------|---------------|
| 0 | `x_position` | 相对于目标平台中心的水平距离 | ✅ |
| 1 | `y_position` | 相对于平台高度（垫面）的垂直距离 | ✅ |
| 2 | `x_velocity` | 水平线速度 | ✅ |
| 3 | `y_velocity` | 垂直线速度 | ✅ |
| 4 | `body_angle` | 机体朝向角（0 表示竖直） | ✅ |
| 5 | `angular_velocity` | 机体旋转角速度 | ✅ |
| 6 | `left_support_contact` | 左支撑腿/触地点与平台接触标志（1.0 接触，0.0 未接触） | ✅ |
| 7 | `right_support_contact` | 右支撑腿/触地点与平台接触标志 | ✅ |

所有观测均可作为奖励信号使用（位置、速度、角度、接触、通过动作可间接约束燃料）。

## 4. 动作空间 action_space
- type: `Discrete`
- n: `4`
- 动作含义：

| action id | 名称 | 含义 |
|-----------|------|------|
| 0 | `no_engine` | 不启动任何引擎（滑行/自由落体） |
| 1 | `left_orientation_engine` | 点燃左侧姿态修正引擎，产生使机体逆时针旋转的力矩 |
| 2 | `main_engine` | 点燃主引擎，沿当前机体方向产生向上推力 |
| 3 | `right_orientation_engine` | 点燃右侧姿态修正引擎，产生使机体顺时针旋转的力矩 |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` 中“settled”部分。当机体稳定着陆在平台上（速度、角速度足够小，且可能通过接触超时判定为 settled），环境会以此条件终止 episode。这是最明确的成功候选。
- **failure-like termination**: 
  - `crash_or_body_contact`：机体与地面或平台以外物体发生不当碰撞（如高速撞击、侧翻触地）。
  - `horizontal_position_outside_viewport`：水平位置超出屏幕边界。
- **ambiguous termination**: `body_not_awake_or_settled` 中的“not_awake”可能导致因坠落/卡死等原因的提前终止，不保证一定成功。
- **truncation**: 从 step 源码未见时间上限截断（`False` 表示无截断），但可能隐含在其他逻辑中；本次不依赖。

### 5.2 success/failure 信号可用性
- `explicit_success_flag_available`: **false**（`info` 为空，未提供显式成功标志）
- `explicit_failure_flag_available`: **false**（同上）
- `allowed_info_fields`: `{}`（当前无任何 info 字段可用）
- `forbidden_or_uncertain_info_fields`: 所有 info 字段均不可用，reward 函数只能使用 `obs`、`action`、`next_obs`（以及可能允许的 `training_progress`，需根据具体 prompt 确定）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs` – 当前观测（8 维）
- `action` – 当前动作（0~3）
- `next_obs` – 下一时刻观测（8 维）
- `info` – 当前 info 为空，**禁止使用任何 info 字段**
- `training_progress` – 仅当 prompt 明确声明允许使用时才能使用

禁止使用：
- `original_reward` – 已屏蔽，**严禁直接或间接使用**
- 任何未在上述列出的 info 字段
- 任何未在观测说明中声明的 obs 切片

## 7. 可用于奖励函数的信号
- **位置信号**: `x_position`，`y_position`（相对于目标平台），可用于表达“距目标多远”。
- **速度信号**: `x_velocity`，`y_velocity`，可用于减速鼓励。
- **姿态信号**: `body_angle`，可用于惩罚倾斜（期望竖直 a≈0）。
- **角速度信号**: `angular_velocity`，可用于平滑控制。
- **接触信号**: `left_support_contact`，`right_support_contact`，可用于奖励安全触地。
- **动作/引擎信号**: `action`，可用于鼓励不加推力（燃料节省）或惩罚引擎使用。

## 8. 不确定或不可用的信号
- **成功/失败标志**: 无显式字段，不能直接读取 episode 是否成功。
- **剩余燃料/引擎限制**: 任务描述中提及“尽量少用引擎”，但未明确硬约束，可能隐含在 episode 长度或引擎持续工作限制中；当前环境未暴露燃料值，不可用。
- **平台中心绝对坐标**: 观测已给出相对位置，原生可用。
- **时间/步数**: 未在观测或 info 中提供，不可用（`training_progress` 需谨慎）。
- **官方奖励**: 不可用，禁止。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: lander (2D, two-legged)
  actuator_type: one_main_engine + two_orientation_engines
  contact_structure: two_leg_contact (left/right flags)
primary_objectives:
  - land softly on the central target pad (near-zero position error, low velocities, low angular velocity, both legs in contact)
  - reach the landed state as quickly as possible (implicit via episode length)
secondary_objectives:
  - minimize engine usage (fire as little as possible, especially when near target)
main_failure_risks:
  - crashing into ground or sides at high speed
  - drifting out of horizontal bounds
  - tipping over after initial contact due to residual angular velocity
  - overly conservative hovering wasting time/episode length
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: goal_distance_encouragement**
  - purpose: 引导机体向目标平台中心移动。
  - why_required: 没有位置引导，agent 很难学会靠近平台。
  - usable_signals: `x_position`, `y_position`
  - risks: 过度奖励可能鼓励高速撞击；需与速度/接触信号联合使用。

- **role_id: soft_landing_velocity_penalty**
  - purpose: 要求接近目标时速度（线速度、角速度）小而稳定。
  - why_required: 安全着陆依赖于低冲击速度，尤其在即将接触时。
  - usable_signals: `x_velocity`, `y_velocity`, `angular_velocity`
  - risks: 过早惩罚速度会阻止探索接近目标的路径，需要随距离或接触条件调节强度。

- **role_id: upright_orientation_incentive**
  - purpose: 保持机体竖直（body_angle≈0），以确保主引擎推力方向正确，着陆稳定。
  - why_required: 倾斜过大会导致侧向漂移、侧翻，主引擎效率下降。
  - usable_signals: `body_angle`
  - risks: 角度惩罚过大会限制必要的小幅度调整，影响水平移动能力。

- **role_id: contact_reward**
  - purpose: 奖励两腿同时稳定接触平台。
  - why_required: 着陆成功的最终标志是接触且未因冲击弹起/翻倒；缺少接触信号则难以判断着陆完成。
  - usable_signals: `left_support_contact`, `right_support_contact`
  - risks: 只给接触奖励而不考虑速度可能导致 agent 硬着陆碰触
