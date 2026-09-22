# Prompt Record

## System Prompt

```text
你是奖励函数生成模块。你将直接读取：
1. environment_card.md：环境事实、任务画像、奖励职责拆解、职责-信号映射；
2. expert_reward_context.md：固定专家 Schema，包括任务类型示例和 Formula Operator Library；
3. optional masked_step_source：默认不提供，除非调试开启。

你的任务不是机械选择某个 skeleton，而是：
1. 读取 environment_card.md 的 `expert_task_profile`；
2. 读取 `reward_role_decomposition`，明确 mandatory / conditional / avoid roles；
3. 使用 `role_to_signal_mapping` 检查每个职责可用的 obs/action/info 信号；
4. 从 expert_reward_context.md 的 Formula Operator Library 中为每个 selected role 选择数学形式；
5. 生成第一版奖励函数 `reward_v1.py`，并附带简短设计说明。

# Expert Schema 使用规则

- environment_card.md 中的 `reward_role_decomposition` 优先级高于 expert_reward_context.md 的模板。
- expert_reward_context.md 只提供专家模板和公式算子，不是固定答案。
- 先选 role，再选 signal，再选 formula operator，最后才写代码。
- 如果某个 role 没有可用信号，必须放入 excluded_roles，不得硬写。
- 如果 task_profile 与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不允许因为模板里提到某个 role 就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康/安全约束；效率、能耗、复杂门控和动态权重默认后续迭代再加入。

# 总体设计原则

- 从简单到复杂，但”简单”不等于只有一个组件。
- 不要用”最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
- 写完 reward 后自检：① 每个终止条件是否有前兆软信号？② 任务目标是否有直接的进度信号？③ 动作维度 ≥ 6 时，是否缺少效率约束（即使权重很小）？
- 不要机械照抄 expert template 或 formula operator。
- 不要使用 original_reward。
- 不要计算 fitness_score 或 fitness_score components。
- 不要使用未声明的 info 字段，例如 info["success"]、info.get("success")。
- 不要使用未声明的 obs 切片，例如 obs[0:3]。
- 只能使用 environment_card.md 声明的观测维度和索引，不得自行扩展为未声明的二维、三维或其他结构。
- 如果 explicit_success_flag_available=false，不要把 terminal_success_reward 写成 v1 核心项。
- 如果 explicit_failure_flag_available=false，不要把 terminal_failure_penalty 写成 v1 核心项。
- 允许使用 obs 和 next_obs 的逐 index 变量。
- 尽量让奖励平滑；需要距离、速度等连续项时，优先使用连续函数。
- 如果需要 sqrt，禁止 import numpy，使用 `** 0.5`。
- 如果想使用 exp 形式的平滑变换，禁止 import numpy；可以使用 `2.718281828 ** (...)`，并显式写 temperature 参数。

# 任务无关设计原则

## 原则 1：信号可用性优先

- 先检查 environment_card.md 中声明的可用信号、禁止信号和 role_to_signal_mapping。
- 只有当信号确实存在于环境接口中时，才设计依赖该信号的组件。
- 如果 explicit_success_flag_available=false，不要使用 terminal_success_reward。
- 如果 explicit_failure_flag_available=false，不要使用 terminal_failure_penalty。
- 不要发明未声明的 info 字段或 obs 切片。

## 原则 2：稠密性

- 优先选择每步都能提供有意义梯度的连续信号。
- 二值条件信号触发率过低时等于摆设。
- 连续函数、bounded 函数、soft proxy 通常比硬阈值更利于学习。

## 原则 3：尺度与平衡

- 不同组件的量级应大致可比，不要让一个组件在数值上统治其他组件。
- 约束/惩罚不应无条件压制任务驱动力；具体尺度必须结合触发频率、数学形态和预期行为判断。
- 差分信号、持续状态奖励和稀疏事件奖励具有不同时间语义，不能仅凭步均值比例判断谁更重要。

## 原则 4：信号冲突

- 不要同时大权重使用两个计算同一物理量的信号。
- 不要让惩罚项压制探索；过严姿态/速度/动作约束可能导致 agent 不敢行动。
- soft_health_gate 比强全局惩罚更适合处理“前进但失稳”的早期问题。

## 原则 5：阶段条件

- v1 阶段避免过早引入效率/动作代价；agent 应先学会任务方向，再优化效率。
- 复杂门控、动态课程、强能耗项默认后续迭代再加入。
- curriculum_weighting 只有当 training_progress 明确允许且任务确有阶段性冲突时才使用。

## 原则 6：可利用风险

- 每个组件都要考虑 agent 可能找到的捷径。
- 只奖励速度可能导致 velocity_burst_then_fall。
- 只奖励存活可能导致 stand_still 或 hover。
- 只奖励接触可能诱导 contact reward hacking。
- 直接奖励 vertical activity 可能诱导原地弹跳。

# role-based component budget

v1 推荐使用 2~4 个组件，按以下角色组织。专家模板和公式算子只提供设计启发，不限制你组合、变形或创造适合当前环境的新信号。

## 必须包含

**1 个主学习信号。** 这是 reward 的核心驱动力，告诉 agent “做什么能得分”。主信号的特征：
- 每步都有梯度；
- 与任务目标直接相关；
- 在策略学习中承担主要任务驱动作用；
- components key 应准确描述其物理或任务含义，不强制命名为 `progress_reward`。

## 允许包含（按需，不是必须全加）

- **0~2 个稳定/安全/健康约束。** 如果任务需要控制速度、姿态、身体高度、角速度等，可以加入轻量惩罚或 soft gate。约束的角色是“方向盘”而非“刹车”。
- **0~1 个任务完成近似信号。** 如果环境没有显式 success flag 但需要在 agent 接近完成时给予额外引导，可以用多条件组合的 soft proxy。proxy 必须由多个连续条件组合，不能直接伪造 success flag。
- **0~1 个效率/动作代价。** v1 默认不加或极小权重；能耗优化通常留到后续迭代。

## 默认不在 v1 使用

- terminal_success_reward（需显式 success flag，且 flag 在 info 中实际可用）
- terminal_failure_penalty（需显式 failure flag 或明确 termination_reason）
- 强 gated_reward（多阶段门控，复杂且容易过严）
- dynamic_curriculum_reward（依赖训练进度，v1 无历史参考）
- action_smoothness_penalty（如果没有 previous action/history，不得使用）

# 输出格式要求

函数签名必须完全一致：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

最终 reward 函数输出必须包含：
1. total_reward: float
2. components: dict，记录 individual reward components

首选返回格式：
```python
return float(total_reward), components
```

# 代码硬约束

- Python code block 里只能包含完整的 `compute_reward` 函数。
- 不要写 import。
- 不要写 class。
- 不要写 try/except。
- 不要写 eval/exec/open。
- 不要创建额外函数。
- 不要引入新的输入变量。
- 不要传 self；当前项目接口不是 Eureka 原版 self 接口。
- 不要使用 self attributes。
- 不要使用原始环境 reward。
- components 必须是 dict。
- components 只包含被加到 total_reward 的组件（A、B、C），不包含 total_reward 本身。

# Markdown 输出要求

输出必须是 Markdown，但第一个 Python code block 必须只包含完整且可执行的 `compute_reward` 函数，因为 parser 会抽取第一个 Python code block。

格式：

# reward_v1.py

```python
def compute_reward(...):
    ...
```

# reward_v1 设计说明

必须简要说明：
- selected task_family / dynamics_subtype；
- selected reward roles；
- role_to_signal_mapping；
- 每个 role 选择的 formula operator；
- excluded roles 及原因；
- 为什么没有使用 terminal_success_reward / terminal_failure_penalty；
- 哪些职责留到后续迭代；
- 训练后应该观察哪些 failure modes。

```

## User Prompt

```markdown
# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个2D飞行器轨迹优化任务。一个带有双支撑腿的机体从视口顶部中央附近开始，具有随机初始力。主要目标是使机体尽快到达视口中央的目标着陆垫，并在其上稳定停靠，同时尽量少使用引擎推力。智能体应学会**接近目标 → 减速 → 保持竖直姿态 → 实现平稳、安全的接触并最终静止**。附属目标包括节省燃料和快速完成，但二者服务于主要目标，不与主要目标构成对等冲突的多目标结构。

## 2. 任务类型选择
**selected_route_id:** `navigation_goal_reaching`  
**confidence:** high  
**reason:** 任务的核心要求是“到达并停靠在中央目标垫上”，观察空间包含相对目标的连续位置，目标区域明确，主要的成功条件与位置、速度、姿态有关，符合导航与目标到达任务族。附属的省燃料、快速性属于典型的性能优化，不改变主任务性质。

## 3. 观察空间 observation_space
- **type:** Box（连续向量）
- **shape:** (8,)
- **dtype:** 根据环境实现推断为 float32
- **各维含义列表：**
  - obs[0]: **x_position** – 机体相对于目标垫中心的水平坐标（纵向），奖励可用：true
  - obs[1]: **y_position** – 机体相对于目标垫高度的垂直坐标（高度），奖励可用：true
  - obs[2]: **x_velocity** – 水平线速度，奖励可用：true
  - obs[3]: **y_velocity** – 垂直线速度，奖励可用：true
  - obs[4]: **body_angle** – 机体倾角（弧度），0 代表竖直向上，奖励可用：true
  - obs[5]: **angular_velocity** – 角速度，奖励可用：true
  - obs[6]: **left_support_contact** – 左支撑腿是否与目标垫接触（1.0=接触，0.0=未接触），奖励可用：true
  - obs[7]: **right_support_contact** – 右支撑腿是否与目标垫接触（1.0=接触，0.0=未接触），奖励可用：true

所有八个维度均可作为奖励信号的来源。

## 4. 动作空间 action_space
- **type:** Discrete
- **n:** 4
- **动作含义：**
  - action 0: **no_engine** – 不点火，任机体惯性运动和受重力/外力影响
  - action 1: **left_orientation_engine** – 点燃左侧姿态发动机，产生使机体逆时针旋转的力矩（同时可能产生微小推力）
  - action 2: **main_engine** – 点燃主发动机，产生向上的推力用于减速或升力
  - action 3: **right_orientation_engine** – 点燃右侧姿态发动机，产生使机体顺时针旋转的力矩（同时可能产生微小推力）

动作空间为离散的四选一，无连续量。

## 5. step 与终止条件分析
### 5.1 终止模式
根据提供的 `termination_conditions`，有三种情况会导致 `terminated = True`：

1. **crash_or_body_contact** — 机体碰撞地面或底部结构，典型失败。
2. **horizontal_position_outside_viewport** — 机体水平位置超出视口边界，典型失败。
3. **body_not_awake_or_settled** — 机体不再“活跃”（速度、角速度极低，且可能已达到稳定状态）。该条件代表物理稳定，可能是成功（若停在目标垫上且姿态好），也可能是失败（若偏离目标或倒在地上）。

由于成功和失败均可能触发 `body_not_awake_or_settled`，且官方 step 没有提供任何额外 success 标志（info 为空字典），因此**不存在显式的成功或失败信号**，仅靠该终止条件不能直接区分成功/失败。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available:** false
- **explicit_failure_flag_available:** false
- **allowed_info_fields:** 无（`info` 返回空字典 `{}`，无额外字段）
- **forbidden_or_uncertain_info_fields:** info 中不存在任何可依赖字段，禁止假设存在 `success`、`failure`、`termination_reason` 等字段。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用：**
- `obs`：当前步的观察向量（8维）
- `action`：选取的动作（0,1,2,3）
- `next_obs`：执行动作后的观察向量（8维）
- `info` 中明确允许的字段：无（因为该环境 info 恒为空）
- `training_progress`：仅当 prompt 明确允许时才可用，此处未提及，默认不建议依赖

**禁止使用：**
- `original_reward`（官方已掩码）
- 任何未在任务描述中声明的 info 字段
- 任何未在观察空间中声明的数据维度

## 7. 可用于奖励函数的信号
- **位置信号：** `next_obs[0]` (x)、`next_obs[1]` (y) — 接近目标时绝对值应趋于零。
- **速度信号：** `next_obs[2]` (x_vel)、`next_obs[3]` (y_vel) — 目标附近速度应接近零。
- **姿态信号：** `next_obs[4]` (body_angle) — 竖直姿态时角度为0（或接近0），可惩罚大偏离。
- **角速度信号：** `next_obs[5]` (angular_velocity) — 目标附近应接近零。
- **接触信号：** `next_obs[6]` (左接触)、`next_obs[7]` (右接触) — 可表征着陆状态，用于鼓励平稳接触。
- **动作/引擎信号：** `action` — 可用于鼓励少用引擎（No.0 无引擎），惩罚使用主引擎或姿态引擎以节省燃料。
- **其他：** 可通过 `obs` 与 `next_obs` 的差分计算位置变化、速度变化等。

## 8. 不确定或不可用的信号
- **环境提供的显式成功/失败标志：** 不存在。
- **目标垫位置：** 默认认为目标垫中心位于 (x=0, y=0)，但并未显式给出，仅由 obs[0] 和 obs[1] 是相对于目标的偏移推断。若实际目标位置有偏移，则不成立，但可安全假定相对坐标已归零。
- **机体质量、推力大小等物理参数：** 不在观察内，不可用。
- **风或其他外部干扰：** 源码中提及 wind 但屏蔽，不可用。
- **全局时间步数：** 不可用，除非 training_progress 被允许，否则不建议在奖励中引入基于时间/步数的奖励。
- **termination 原因判别：** 无法从 info 获得，禁止假设某终止条件对应成功。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact   # 接近目标并低速、稳定接触/停靠
control_type: discrete                             # 动作为离散四选一
morphology:
  body_type: 2D_lander_with_two_legs               # 双支撑腿着陆器
  actuator_type: main_engine + left/right_orientation_engines
  contact_structure: two_point_contact (left_leg, right_leg)
primary_objectives:
  - arrive_at_target_pad: minimize final distance to (x=0, y≈目标垫高度)，安全停靠
  - stable_settlement: final velocities and angular velocity near zero, upright orientation
secondary_objectives:
  - fuel_efficiency: minimize engine usage (non-zero actions)
  - time_efficiency: reach the pad quickly (隐含，但无直接步数信号可用)
main_failure_risks:
  - crashing_into_ground: body contact with ground or other obstacles
  - drifting_out_of_viewport: horizontal position beyond bounds
  - unstable_landing: high velocity contact or tilted body causing failure even after pad touch
  - overshoot_or_undershoot: poor approach dynamics leading to failure before settlement
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id:** `distance_to_target`
  - **purpose:** 引导飞行器向目标垫移动，每一步缩小相对距离。
  - **why_required:** 主要目标是到达目标位置，没有位置引导将无法收敛到垫附近。
  - **usable_signals:** `next_obs[0]` (x), `next_obs[1]` (y)
  - **risks:** 单纯惩罚距离可能忽略速度控制，导致以高速撞击垫或超出；需要配合速度与姿态惩罚。

- **role_id:** `soft_landing_velocity`
  - **purpose:** 在接近目标时降低线速度和角速度，实现平稳停靠。
  - **why_required:** 若仅引导位置，飞行器可能以高速撞击垫，触发 crash 或因不稳定导致失败，无法完成“稳定停靠”。
  - **usable_signals:** `next_obs[2]`, `next_obs[3]`, `next_obs[5]` (x_vel, y_vel, angular_vel)
  - **risks:** 过早施加强力速度惩罚可能阻止飞行器从高处下落，需在近距离或接触附近加强。

- **role_id:** `upright_orientation`
  - **purpose:** 保持机体竖直，防止因姿态过大导致碰撞或无法稳定。
  - **why_required:** 任务明确要求“保持稳定姿态”；倒置或大倾角接触垫通常导致失败。
  - **usable_signals:** `next_obs[4]` (body_angle)
  - **risks:** 角度稳定性可能需配合角速度惩罚一同使用。

### 10.2 条件职责 conditional_roles
- **role_id:** `fuel_penalty`
  - **purpose:** 鼓励减少引擎使用，节省燃料。
  - **condition_to_use:** 当飞行器已足够接近目标（例如 |x|<阈值、y 较低）或者接触垫后加强，早期可不施加或极轻微，避免干扰下降探索。
  - **usable_signals:** `action` (非0动作受惩罚)
  - **risks:** 若全程施加较强的引擎惩罚，可能导致飞行器不敢点火而直接坠落失败，必须与距离和速度职责平衡。

- **role_id:** `contact_reward`
  - **purpose:** 奖励双腿接触垫并保持稳定，强化安全着陆行为。
  - **condition_to_use:** 当 `next_obs[6]` 和 `next_obs[7]` 均为 1.0 时给予一次性或持续小奖励；也可结合速度低、角度小来提升奖励可靠性。
  - **usable_signals:** `next_obs[6]`, `next_obs[7]`, 速度、角度
  - **risks:** 如果单纯奖励接触而不检查速度和角度，可能鼓励猛烈撞击，应组合使用。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id:** `time_step_penalty`
  - **reason:** 无法获取全局步数，training_progress 未被明确允许使用；即使可用，也不应在无训练进度参数时依赖。
  - **forbidden_or_missing_signals:** 全局时间步数。

- **role_id:** `success_oriented_bonus`
  - **reason:** 环境无显式成功标志，无法在奖励函数中区分成功终止。
  - **forbidden_or_missing_signals:** info 中无 success 字段。

- **role_id:** `exact_zoning_reward` (如同环数奖励)
  - **reason:** 目标垫周围无离散区域划分，观察无相关信息，不适用。
  - **forbidden_or_missing_signals:** 区域编号。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| `distance_to_target` | `next_obs[0]`, `next_obs[1]` | 无 | `dense_state_signal`, `bounded_signal` (e.g., -L2距离) | 可对水平和垂直分量分别加权，因垂直下降可能需单独引导。 |
| `soft_landing_velocity` | `next_obs[2]`, `next_obs[3]`, `next_obs[5]` | 无 | `quadratic_penalty` (速度惩罚), `bounded_signal` | 建议在接近垫或接触时增强，可与接触标志联动。 |
| `upright_orientation` | `next_obs



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





# Fresh Restart Evidence

- target_score: 200.000
- best_score_so_far: 5.670

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| proximity_reward + safe_contact_bonus + soft_landing_penalty | 4 | 5.670 | -9.340 | unsolved |

## Previous interventions

- No structured intervention fields were available in the historical responses.

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.

```
