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
- 写完 reward 后自检：① 每个终止条件是否有前兆软信号？② 任务目标是否有直接的进度信号？③ 动作维度 ≥ 6 时，是否缺少效率约束（即使权重很小）？④ **环境卡片最后一节的每一条 failure mode 是否都有对应组件在防止它？**
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

# 必须落实的 failure modes（写代码前先做这一步）

`environment_card.md` 的最后一节列出了**本环境已知的 failure modes**，每一条都带有 `evidence_to_check` 和 `possible_intervention`。

**这不是"训练以后再回头看"的观察清单，而是本版奖励必须处理的设计要求。** 先逐条过这张表，再决定组件。

对每一条 failure mode，你必须在设计说明里回答：**它由哪个具名组件、通过什么机制被防止？** 若你的设计无法防止某一条（例如所需信号不可用），必须显式写出原因，并把它放进 excluded_roles。

以下几条是硬性要求，违反即视为未完成：

1. **过冲 / 冲过目标 / 无法停稳**
   若表中出现"推进过快、冲过目标、无法停稳、无法减速"一类条目，**只奖励"更接近目标"就是无效设计**——它本身就会产生这个 failure mode。
   你必须引入一个**在接近目标时抑制速度的机制**，例如：用低速因子对进度信号加权或门控；或在接近目标时施加速度 hinge 惩罚。
   注意：该机制不得阻断早期探索（远离目标时不应把奖励压到 0）。

2. **停在中间状态刷分 / proxy 被悬停收割**
   若表中出现这一类条目，主信号必须使用**改善量（delta）**，不能用**状态值（proximity）**。占据某个状态本身不得持续得分。

3. **卡在狭窄通道 / 障碍**
   若表中出现这一类条目，需要提供通过该处的引导信号或局部惩罚，而不是只靠全局进度信号。

4. **接触 / 冲击类**
   若表中出现硬冲击或易碎约束条目，必须有对应的冲击或相对速度惩罚，且只在接触时生效。

5. **辅助组件不得与任务推进方向对抗（可机检，违反即无效）**
   任何辅助组件（低速、稳定、对齐、平滑等）都**不允许因为"推进任务所必需的动作"而下降**。

   典型反例：用 `k / (1 + c * speed)` 形式的"低速因子"作为**持续的全局状态奖励**。推动目标物体必然产生速度，于是该组件在惩罚任务进展本身；一旦它的量级压过主信号，最优策略就变成"什么都不做"。

   要求：
   - "低速 / 静止 / 对齐"这类量**只能作为门控**——即乘在任务进展信号上，或仅在**已经接近目标**时激活；**不得作为全局持续奖励**被动收分；
   - 任何组件都不允许对"把目标推向目标位置"这一行为产生净负贡献。

   **必做自检（必须通过才能输出）**：把你的奖励函数分别代入下面两种状态，计算单步总奖励：
   - 状态 ①：智能体什么都不做，目标物体静止在**初始位置**；
   - 状态 ②：智能体正在把目标物体**推向目标位置**（目标物体因此具有速度）。
   要求 **状态 ② 的单步总分严格高于状态 ①**。若做不到，说明存在激励冲突，必须重新设计——不要输出未通过该自检的版本。

   在设计说明中必须给出这两次代入的**具体数值**。

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
- **failure mode 覆盖表**（必须有，格式如下）：

```
| failure_mode | 对应组件 | 防止机制 |
|---|---|---|
| （抄环境卡片最后一节的条目） | （组件名，或"未覆盖") | （一句话说明它如何防止该模式） |
```

Important: the FIRST Python code block must contain ONLY the complete `compute_reward` function, starting with `def compute_reward(` and ending with `return ...`. Do not write any text before it. The short design section, including the mandatory failure-mode coverage table, goes AFTER the code block.
```

## User Prompt

```markdown
# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个俯视视角的仓库推箱任务：一辆无刹车、无夹爪的轮式小车，需要把一只**易碎**的方形货箱从仓库近侧推到隔墙另一侧的**交付泊位**内，并让货箱在泊位中**完全进入、朝向对齐、几乎静止**并持续一小段时间。主目标是“把货箱安全送达并稳定停靠”；次目标是“避免货箱受到硬冲击损坏”“避免小车或货箱离开仓库地面”“在时间预算内完成”。**不该混淆的目标**：单纯“靠近泊位”“触碰泊位区域”“把货箱推得越快越好”都不是成功；由于小车无法从后方减速货箱，把“持续推到底”当作目标会与低速停靠条件冲突。

## 2. 任务类型选择
selected_route_id: manipulation_grasping
confidence: medium
reason: 核心是“把物体（货箱）移动到指定位姿（泊位内、朝向对齐、静止）”，属于对物体的操控与位姿达成，而非单纯导航或持续前进。虽然载体是轮式小车、动作是驾驶式连续控制，但成功判据完全落在**货箱的位姿与速度**上，且存在“接近—推入—稳定停靠”的阶段结构，因此归为 manipulation_grasping 最贴切。次目标（不损坏、不出界、限时）是约束而非并列主目标，故不选 multi_objective_task。

## 3. 观察空间 observation_space
- type: Box
- shape: [19]
- dtype: float32
- bounds: 所有分量裁剪到 [-2.0, 2.0]
- obs[0]: cart_x，小车 x 位置 / 仓库半宽（0=中线，+1=远墙），reward_usable: true
- obs[1]: cart_y，小车 y 位置 / 仓库半高（0=中线），reward_usable: true
- obs[2]: cart_cos_heading，小车朝向余弦，reward_usable: true
- obs[3]: cart_sin_heading，小车朝向正弦，reward_usable: true
- obs[4]: cart_forward_speed，小车沿自身朝向速度 / 3.0 (m/s)，reward_usable: true
- obs[5]: cart_yaw_rate，小车角速度 / 8.0 (rad/s)，reward_usable: true
- obs[6]: crate_rel_x_body，货箱相对小车在车体系 x 分量 / 3.0 (m)，reward_usable: true
- obs[7]: crate_rel_y_body，货箱相对小车在车体系 y 分量 / 3.0 (m)，reward_usable: true
- obs[8]: crate_vx，货箱世界系线速度 x / 3.0 (m/s)，reward_usable: true
- obs[9]: crate_vy，货箱世界系线速度 y / 3.0 (m/s)，reward_usable: true
- obs[10]: crate_cos_heading，货箱朝向余弦，reward_usable: true
- obs[11]: crate_sin_heading，货箱朝向正弦，reward_usable: true
- obs[12]: crate_to_dock_x，货箱中心到泊位中心的有符号 x 偏移 / 仓库半宽，reward_usable: true
- obs[13]: crate_to_dock_y，货箱中心到泊位中心的有符号 y 偏移 / 仓库半高，reward_usable: true
- obs[14]: cart_crate_contact，小车与货箱当前是否接触（1/0），reward_usable: true
- obs[15]: sensor_front，小车前方最近静态障碍接近度（0=远，1=接触），reward_usable: true
- obs[16]: sensor_left，小车左侧最近静态障碍接近度，reward_usable: true
- obs[17]: sensor_right，小车右侧最近静态障碍接近度，reward_usable: true
- obs[18]: time_fraction，已消耗时间预算比例 [0,1]，reward_usable: true

## 4. 动作空间 action_space
- type: Box（连续）
- shape: [2]
- bounds: 每通道 [-1.0, 1.0]
- action[0]: drive，沿小车朝向的纵向力指令；+1 前进，-1 倒车
- action[1]: steer，转向力矩指令；+1 左转（逆时针），-1 右转

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `docked_success` —— 货箱完全在泊位内、朝向误差 < 30°、速度 < 0.05 m/s，且连续保持 10 个环境步。
- failure-like termination: `crate_out_of_bounds`（货箱中心离开仓库地面）、`cart_out_of_bounds`（小车中心离开仓库地面）、`crate_damaged`（硬冲击计数 ≥ 3）。
- ambiguous termination: 无显式歧义终止；但“货箱进入泊位但未满足朝向/速度/持续条件”不会终止，属于未完成状态。
- truncation: `time_limit` —— 达到固定步数预算，报告为截断，**不是成功**。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 中 `is_success` 被列为 forbidden）
- explicit_failure_flag_available: false（`termination_reason`、`hard_collision_count` 等被 forbidden）
- allowed_info_fields: []（无任何允许读取的 info 字段）
- forbidden_or_uncertain_info_fields: is_success, cargo_goal_distance, cargo_angle_error, cargo_speed, robot_cargo_distance, contact_impulse, hard_collision_count, stagnation_steps, action_energy, component_returns, official_reward_terms, termination_reason, cargo_inside_dock, stable_steps

> 说明：成功/失败只能通过观测信号**间接推断**（derived_possible），不能直接读取 info。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测，19 维）
- action（当前动作，2 维）
- next_obs（下一步观测，19 维）
- info 中明确允许的字段：**无**（allowed_info_fields 为空）
- training_progress：仅当 prompt 明确允许时才用；本环境未声明允许，默认不使用

禁止使用：
- original_reward（官方奖励被 mask）
- official_reward / masked_reward
- 任何未声明的 info 字段（尤其 5.2 列出的 forbidden 字段）
- 未声明的 obs 切片（只能使用 0–18 已声明维度）

## 7. 可用于奖励函数的信号
- position:
  - 小车世界位置：由 obs[0]*5.0, obs[1]*4.0 恢复（仓库半宽 5.0，半高 4.0）。
  - 货箱世界位置：将 (obs[6]*3.0, obs[7]*3.0) 按小车朝向旋转后加上小车位置。
  - 货箱到泊位偏移：obs[12]*5.0, obs[13]*4.0（直接可用）。
- velocity:
  - 小车前向速度：obs[4]*3.0。
  - 小车角速度：obs[5]*8.0。
  - 货箱世界速度：obs[8]*3.0, obs[9]*3.0；货箱轴向速率 = sqrt((obs[8]*3.0)^2 + (obs[9]*3.0)^2)。
- orientation:
  - 小车朝向：atan2(obs[3], obs[2])。
  - 货箱朝向：atan2(obs[11], obs[10])；货箱朝向误差可结合泊位朝向推断（derived_possible）。
- contact:
  - 小车-货箱接触：obs[14]（1/0）。
  - 静态障碍接近度：obs[15]/obs[16]/obs[17]（隔墙与开口为硬障碍）。
- action/engine:
  - drive=action[0]，steer=action[1]；可用于动作平滑/能耗类信号（若任务要求）。
- other:
  - time_fraction=obs[18]，可用于时间相关调度或截断前行为调整。
  - derived_possible 推断路径：
    - 成功接近：货箱到泊位偏移 (obs[12], obs[13]) 持续减小且货箱速率趋近 0。
    - 货箱损坏风险：obs[14]=1 且小车前向速度 obs[4] 较大（高冲击代理），或货箱速度突变（derived_possible）。
    - 出界：小车/货箱世界坐标超出合理范围（derived_possible）。
    - 卡住/停滞：货箱到泊位偏移长时间不减小（derived_possible）。

## 8. 不确定或不可用的信号
- 官方奖励 masked_reward：不可用。
- info 全部字段：不可用（含 is_success、termination_reason、hard_collision_count、contact_impulse、cargo_inside_dock、stable_steps 等）。
- 精确的“硬冲击计数”“峰值法向冲量”：不可直接读取，只能用 obs[14] 与速度量做代理（derived_possible，噪声大）。
- 泊位矩形的精确边界与朝向：未在 obs 中显式给出，只能通过 obs[12]/obs[13] 的偏移与货箱朝向间接推断（derived_possible）。
- 隔墙开口的精确几何：未显式给出，只能通过 obs[15]/obs[16]/obs[17] 接近度间接感知。
- 货箱“完全进入泊位”的布尔量：不可直接读取，需由偏移与货箱尺寸推断（derived_possible）。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: manipulation_grasping
dynamics_subtype: staged_manipulation
control_type: continuous
morphology:
  body_type: wheeled_cart_top_down
  actuator_type: longitudinal_force_plus_steering_torque
  contact_structure: rigid_body_contact_no_gripper_push_only
primary_objectives:
  - 将易碎货箱推入泊位并使其完全进入、朝向对齐、几乎静止并持续保持
secondary_objectives:
  - 避免货箱受到硬冲击（易碎约束）
  - 避免小车或货箱离开仓库地面
  - 在时间预算内完成
main_failure_risks:
  - 硬冲击累计达 3 次导致货箱损坏
  - 小车或货箱出界
  - 货箱进入泊位但速度/朝向不满足，无法稳定停靠
  - 小车无刹车，货箱滑行过头冲出泊位
  - 隔墙开口狭窄，推箱路径被卡住或撞墙
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: crate_to_dock_progress
  purpose: 驱动货箱向泊位中心靠近（接近阶段主信号）。
  why_required: 任务核心是把货箱移动到泊位，必须有指向泊位的进度信号。
  usable_signals: [obs[12], obs[13], obs[6], obs[7], obs[0], obs[1], obs[2], obs[3]]
  risks: 若用“接近度”而非“增量”，agent 可能停在泊位附近不完成；需配合停靠条件。
- role_id: crate_settle_and_align
  purpose: 在货箱接近泊位后，鼓励其低速、朝向对齐并稳定停靠。
  why_required: 成功判据要求货箱完全进入、朝向对齐、速度 < 0.05 m/s 并持续 10 步。
  usable_signals: [obs[8], obs[9], obs[10], obs[11], obs[12], obs[13]]
  risks: 过早强调低速会抑制接近进度；需按阶段或按接近程度条件化。

### 10.2 条件职责 conditional_roles
- role_id: fragile_impact_avoidance
  condition_to_use: 当 obs[14]=1（接触）且小车前向速度 obs[4] 较大时，抑制高冲击推撞。
  usable_signals: [obs[14], obs[4], obs[8], obs[9]]
  risks: 无精确冲量信号，只能用速度代理，可能误罚正常推箱。
- role_id: boundary_avoidance
  condition_to_use: 当小车或货箱接近仓库边界（由世界坐标推断）时加入。
  usable_signals: [obs[0], obs[1], obs[6], obs[7], obs[2], obs[3]]
  risks: 边界几何未显式给出，需谨慎设定阈值。
- role_id: obstacle_avoidance
  condition_to_use: 当 obs[15]/obs[16]/obs[17] 接近 1（接近隔墙/障碍）时加入。
  usable_signals: [obs[15], obs[16], obs[17]]
  risks: 可能抑制必要的穿墙开口操作，需与开口通过行为协调。
- role_id: action_smoothness_or_energy
  condition_to_use: 仅当任务明确要求高效/平滑时加入；本任务描述未强调，默认不加。
  usable_signals: [action[0], action[1]]
  risks: 属附属优化，可能干扰主任务。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: official_reward_shaping
  reason: 官方奖励被 mask，禁止使用。
  forbidden_or_missing_signals: [masked_reward, original_reward, official_reward_terms]
- role_id: info_based_success_bonus
  reason: info 全部字段 forbidden，无法读取 is_success/termination_reason/cargo_inside_dock/stable_steps。
  forbidden_or_missing_signals: [is_success, termination_reason, cargo_inside_dock, stable_steps]
- role_id: hard_collision_penalty_from_info
  reason: hard_collision_count/contact_impulse 被 forbidden，只能用速度代理。
  forbidden_or_missing_signals: [hard_collision_count, contact_impulse]
- role_id: push_until_end
  reason: 小车无刹车，货箱靠地面阻尼滑行；持续推到底会破坏低速停靠条件。
  forbidden_or_missing_signals: []
- role_id: pure_proximity_hover
  reason: 仅用接近度会让 agent 停在泊位附近收割分数而不完成停靠。
  forbidden_or_missing_signals: []

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| crate_to_dock_progress | obs[12], obs[13]（及恢复的世界坐标） | 无 | delta(distance), improvement | 用增量而非纯接近度，避免悬停 |
| crate_settle_and_align | obs[8], obs[9], obs[10], obs[11], obs[12], obs[13] | 泊位精确朝向/边界 | bounded_signal, quadratic_penalty, hinge | 按接近程度条件化，避免抑制接近 |
| fragile_impact_avoidance | obs[14], obs[4], obs[8], obs[9] | contact_impulse, hard_collision_count | hinge, gate | 速度代理，阈值需保守 |
| boundary_avoidance | obs[0], obs[1], obs[6], obs[7], obs[2], obs[3] | 精确边界几何 | hinge, quadratic_penalty | derived_possible，阈值需推断 |
| obstacle_avoidance | obs[15], obs[16], obs[17] | 开口精确几何 | hinge, gate | 避免误罚穿开口行为 |
| action_smoothness_or_energy | action[0], action[1] | 无 | quadratic_penalty | 默认不加，除非明确要求 |
| official_reward_shaping | 无 | masked_reward | — | 禁用 |
| info_based_success_bonus | 无 | is_success 等 | — | 禁用 |
| hard_collision_penalty_from_info | 无 | hard_collision_count | — | 禁用 |
| push_until_end | 无 | 无 | — | 禁用，与停靠条件冲突 |
| pure_proximity_hover | obs[12], obs[13] | 无 | proximity | 慎用，易悬停 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 货箱被推过头冲出泊位 | 货箱到泊位偏移先减小后反向增大；货箱速率在泊位附近仍高 | 在接近泊位时降低推进力度，加入低速/停靠条件化信号 |
| 货箱进入泊位但速度不达标 | 偏移小但货箱速率长期 > 0.05 m/s | 强化低速停靠职责，弱化接近阶段信号 |
| 货箱朝向未对齐 | 货箱朝向误差长期 > 30° | 加入朝向对齐信号，或调整推箱接触点策略 |
| 硬冲击导致货箱损坏 | 接触时小车前向速度大；episode 提前终止 | 加入冲击抑制（hinge/gate），降低推撞速度 |
| 小车或货箱出界 | 世界坐标接近/超出仓库范围 | 加入边界避让信号 |
| 卡在隔墙开口 | 小车/货箱位置长时间不变，接近度传感器高 | 加入障碍避让与开口通过引导 |
| 悬停在泊位附近不完成 | 偏移小但无停靠进展，episode 拖到截断 | 用增量信号替代纯接近度，加入停靠完成条件 |
| 时间耗尽截断 | time_fraction 接近 1 且未停靠 | 调整时间相关调度或提高接近效率 |



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


```
