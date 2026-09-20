# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个俯视（bird's-eye）的仓库推动式操纵任务：一辆带轮小车在平面仓库地板上运动，只能施加沿车头方向的纵向驱动力（油门/倒车）和转向力矩，既没有刹车也没有夹爪；场地中有一个可自由滑动的方形货箱（脆弱、必须轻拿轻放），仓库被一道带唯一窄通道的隔墙分成近侧与远侧，远侧地面上有一个标记好的矩形停靠位（dock）。**主目标**是把货箱推过窄通道并送入停靠位，且满足三个同时成立的条件并保持约 10 个环境步：货箱完全位于停靠矩形内、货箱朝向与停靠位对齐（误差 < 30°）、货箱速度 < 0.05 m/s（近似静止）。**次目标**是全程不损坏货箱（连续 3 次硬碰撞即失败）且不越出仓库地板。**不该混淆的目标**：到达通道口、接触货箱、把货箱推到停靠位附近但仍在滑行、或“尽快结束 episode”都不是成功；停靠位内的高速通过或斜向停放不算完成；由于小车没有刹车，货箱只能靠地面阻尼减速，因此“一直推到最后一刻”的策略与低速沉降条件是冲突的。

## 2. 任务类型选择
selected_route_id: manipulation_grasping
confidence: high
reason: 核心目标是“把外部物体（货箱）移动到指定位置并满足指定位姿/状态”，主体动作通过接触推动实现，属于非抓取式物体操纵；虽然画面是俯视导航式布局，但评价对象是货箱到达 dock 的位姿与速度，而不是智能体自身到达某点，因此不选 navigation_goal_reaching。任务存在明显的阶段结构（接近货箱 → 对准并穿过窄通道 → 把货箱推到 dock 附近 → 让货箱滑行减速并在 dock 内对齐沉降），但这只是实现路径的阶段划分，整体仍是单一主目标，不构成 multi_objective_task（“轻拿轻放”与“低速沉降”是同一主目标下的必要约束，不是与送达目标权重相当的冲突目标）。

## 3. 观察空间 observation_space
- type: Box
- shape: [19]
- dtype: float32
- bounds: 所有分量被裁剪到 [-2.0, 2.0]
- 说明：位置量按仓库半宽/半高归一化（x 除以 5.0 m，y 除以 4.0 m），速度量按 3.0 m/s 归一化，角速度按 8.0 rad/s 归一化，相对位置量按 3.0 m 归一化。

| index | name | meaning | reward_usable |
|---|---|---|---|
| 0 | cart_x | 小车 x 位置 / 仓库半宽 5.0（0=中线，+1=远墙） | true |
| 1 | cart_y | 小车 y 位置 / 仓库半高 4.0 | true |
| 2 | cart_cos_heading | 小车朝向余弦 | true |
| 3 | cart_sin_heading | 小车朝向正弦 | true |
| 4 | cart_forward_speed | 小车沿自身车头方向速度 / 3.0 m·s⁻¹ | true |
| 5 | cart_yaw_rate | 小车角速度 / 8.0 rad·s⁻¹ | true |
| 6 | crate_rel_x_body | 货箱相对小车、车体坐标系 x 分量 / 3.0 m | true（用于反解货箱世界坐标/相对速度） |
| 7 | crate_rel_y_body | 货箱相对小车、车体坐标系 y 分量 / 3.0 m | true |
| 8 | crate_vx | 货箱世界系 x 速度 / 3.0 m·s⁻¹ | true |
| 9 | crate_vy | 货箱世界系 y 速度 / 3.0 m·s⁻¹ | true |
| 10 | crate_cos_heading | 货箱朝向余弦 | true（但见第 8 节：缺少 dock 朝向基准） |
| 11 | crate_sin_heading | 货箱朝向正弦 | true（同上） |
| 12 | crate_to_dock_x | 货箱中心到 dock 中心的有符号 x 偏移 / 5.0 | true（主信号核心） |
| 13 | crate_to_dock_y | 货箱中心到 dock 中心的有符号 y 偏移 / 4.0 | true（主信号核心） |
| 14 | cart_crate_contact | 车-箱是否接触（0/1） | true（仅二值，无冲量幅值） |
| 15 | sensor_front | 车前方最近静态障碍接近度（0=范围内无障碍，1=接触） | true |
| 16 | sensor_left | 车左侧最近静态障碍接近度 | true |
| 17 | sensor_right | 车右侧最近静态障碍接近度 | true |
| 18 | time_fraction | 已消耗时间预算比例 ∈ [0,1] | true（可用但慎用，见第 10 节） |

可直接反解的量：
- 货箱世界坐标 = R(cart_heading) · (obs[6]*3.0, obs[7]*3.0) + (obs[0]*5.0, obs[1]*4.0)
- 货箱到 dock 距离 = sqrt((obs[12]*5.0)² + (obs[13]*4.0)²)
- 货箱速度大小 = sqrt((obs[8]*3.0)² + (obs[9]*3.0)²)
- 车-箱相对速度可由 cart_forward_speed + 车头向量与 crate 世界速度组合得到

## 4. 动作空间 action_space
- type: Box（连续）
- shape / n: [2]
- bounds: 每通道 [-1.0, 1.0]

| action dim | name | meaning |
|---|---|---|
| 0 | drive | 沿小车车头方向的纵向力指令；+1 前进，-1 倒车（⇒ force = heading_vec · drive · MAX_FORCE） |
| 1 | steer | 转向力矩指令；+1 左转（逆时针），-1 右转（⇒ torque = steer · MAX_TORQUE） |

关键动力学约束：**没有独立刹车通道**。要减速只能减小 drive 或反向 drive（会先抵消自身速度），而货箱自身的减速只能靠地板阻尼（线性/角阻尼），小车从后方无法让货箱减速。动作空间中没有夹爪、没有抓取维度，只能通过碰撞推动货箱。

## 5. step 与终止条件分析

### 5.1 终止模式
- success-like termination: `docked_success` —— 货箱完全位于 dock 内 + 朝向误差 < 30°（0.5236 rad）+ 线速度 < 0.05 m/s，并且连续保持 10 个环境步（`stable_steps >= 10`）。这是唯一的成功终止。
- failure-like termination:
  - `crate_out_of_bounds`：货箱中心离开仓库地板矩形；
  - `cart_out_of_bounds`：小车中心离开仓库地板矩形；
  - `crate_damaged`：硬碰撞累计 ≥ 3 次（单次硬碰撞定义为一步内车-箱接触的峰值法向冲量超过脆弱阈值）。
- ambiguous termination: 无。上述四类互斥列出。
- truncation: `time_limit` —— 达到固定步数预算，明确“记为 truncated，不算成功”。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（`is_success`、`cargo_inside_dock`、`stable_steps` 均在禁用列表）
- explicit_failure_flag_available: false（`hard_collision_count`、`contact_impulse`、`termination_reason` 均在禁用列表）
- allowed_info_fields: []（空）
- forbidden_or_uncertain_info_fields: `is_success`, `cargo_goal_distance`, `cargo_angle_error`, `cargo_speed`, `robot_cargo_distance`, `contact_impulse`, `hard_collision_count`, `stagnation_steps`, `action_energy`, `component_returns`, `official_reward_terms`, `termination_reason`, `cargo_inside_dock`, `stable_steps`
- 补充：`compute_reward` 签名里没有 `terminated`/`truncated` 参数，因此**无法直接读取是否终止**，终端事件只能靠 obs 序列间接推断（见第 7、11 节，标注为 derived_possible）。

间接推断路径（derived_possible）：
- 成功终止推断：obs[12]、obs[13] 同时接近 0（货箱到达 dock 中心附近）**且** 货箱速度 sqrt((obs[8]*3)²+(obs[9]*3)²) 接近 0 **且** 货箱朝向与假定 dock 朝向一致。局限：dock 矩形尺寸与朝向未在 obs 中给出，只能近似。
- 越界失败推断：|obs[0]| 或 |obs[1]| 接近/超过 1.0（小车离开地板）；或反解出的货箱世界坐标超出地板矩形。
- 货箱损坏失败推断：obs[14] 由 0→1 的瞬间，车-箱相对接近速度很大（由 obs[4]、obs[8]、obs[9]、obs[2..3] 组合估计）；只能作为冲量代理，误差大。
- 时间截断推断：obs[18] 接近 1.0。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（19 维，按第 3 节定义的全部切片）
- action（2 维：drive、steer）
- next_obs（用于计算步间差分：距离 delta、速度变化、接触跳变 0→1 等）
- info 中明确允许的字段：**没有**（allowed_info_fields 为空）
- training_progress：本环境 prompt 未明确允许，**不使用**

禁止使用：
- original_reward / official_reward（被 mask，且禁止复现官方奖励）
- 任何未声明或被禁用的 info 字段（尤其 `contact_impulse`、`hard_collision_count`、`stable_steps`、`cargo_inside_dock`、`termination_reason`、`is_success`）
- 未声明的 obs 切片（必须严格使用 obs[0..18] 的既有语义）
- 任何外部/全局状态（如上一 episode 的轨迹缓存不算违规，但不得读取环境内部计数器）

## 7. 可用于奖励函数的信号
- position:
  - 小车世界坐标 (obs[0]*5.0, obs[1]*4.0)
  - 货箱世界坐标（由 obs[0..3]、obs[6]、obs[7] 反解，derived_possible 但数学上是精确的）
  - 货箱到 dock 偏移 (obs[12]*5.0, obs[13]*4.0) ⇒ 货箱-停靠点欧氏距离（直接可得）
  - 车-箱相对位置（obs[6]、obs[7]，车体坐标系）
- velocity:
  - 小车前向速度 obs[4]*3.0，小车角速度 obs[5]*8.0
  - 货箱世界速度 (obs[8]*3.0, obs[9]*3.0)，货箱线速度大小
  - 车-箱相对接近速度（由小车速度向量与货箱速度向量之差投影到接触法向估计，derived_possible）
- orientation:
  - 小车朝向 (obs[2], obs[3])；货箱朝向 (obs[10], obs[11])
  - 货箱朝向误差：obs 中**没有** dock 朝向基准，只能假定 dock 与仓库轴对齐（例如 0/90°），再用 atan2(obs[11], obs[10]) 与最近轴向比较（derived_possible，带不确定性）
- contact:
  - obs[14] 车-箱接触二值标志（可检测接触发生与持续时间，不可检测冲量大小）
  - obs[15..17] 前/左/右障碍接近度（用于感知隔墙/地板边界，但只给最近障碍的接近度，无法定位窄通道的开口方向）
- action/engine:
  - action[0] drive、action[1] steer 的原始命令（可用于动作变化率/平滑度；本环境未要求能耗指标）
  - 由 drive 可推算施加的推力方向与大小（大小级），可用于估计“是否在推挤”
- other:
  - obs[18] time_fraction：时间预算进度（用于 pacing/门控，慎用）
  - 步间差分：Δ(货箱到 dock 距离)、Δ(货箱速度)、接触 0→1 跳变、货箱位置净位移

## 8. 不确定或不可用的信号
- 接触冲量峰值 `contact_impulse`：禁用，不可用。obs 只有二值接触，无法精确得知“是否硬碰撞”。硬碰撞只能用“接触瞬间的相对接近速度”做代理（derived_possible，噪声大）。
- 硬碰撞计数 `hard_collision_count`：禁用，不可用。
- `stable_steps`、`cargo_inside_dock`：禁用，不可用。且**dock 矩形的具体尺寸/朝向未在 obs 中给出**，因此“货箱完全位于 dock 内”这一条件无法精确判定，只能用 obs[12]、obs[13] 的小邻域近似。
- `termination_reason`、`is_success`：禁用，不可用；`compute_reward` 也拿不到 done 标志。
- dock 朝向基准：不可直接观测。货箱朝向误差只能按“假定停靠位与仓库坐标轴对齐”近似（derived_possible，置信度 medium）。
- 货箱与隔墙/其它静态物体的接触状态：只能通过 obs[12..17] 间接推断，且传感器只给最近障碍接近度，无法判断是隔墙、地板边界还是 dock 边。
- 货箱质量、地面摩擦/阻尼系数、脆弱冲量阈值：不可观测。
- 内部进度量 `cargo_goal_distance`、`robot_cargo_distance`、`cargo_speed`、`cargo_angle_error`、`stagnation_steps`、`action_energy`、`component_returns`：全部禁用；其中距离、速度、角度可由 obs 自己重建（见第 3 节），stagnation / energy / 分项回报则完全不可用。
- 官方奖励 `original_reward`：被 mask，禁止作为信号或参考。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: manipulation_grasping
dynamics_subtype: staged_manipulation   # 非抓取式推动变体（nonprehensile push delivery）
control_type: continuous
morphology:
  body_type: wheeled_cart_on_planar_floor
  actuator_type: longitudinal_force_plus_steering_torque   # 无刹车、无夹爪
  contact_structure: nonprehensile_cart_crate_contact_plus_static_partition
primary_objectives:
  - 将货箱送入 dock，并同时满足 containment + 朝向对齐（<30°）+ 近静止（<0.05 m/s）并保持约 10 步
secondary_objectives:
  - 全过程避免硬碰撞（累计 3 次即失败）——货箱脆弱，需低速接触
  - 货箱与小车都不离开仓库地板
  - 在时间预算内完成（超时为 truncation，不算成功）
main_failure_risks:
  - 推得过猛，货箱滑行进入 dock 时超速/斜向，无法满足低速沉降与朝向对齐
  - 车-箱硬碰撞累计 3 次导致直接失败
  - 找不到隔墙窄通道，长期在近侧徘徊直至超时
  - 货箱被推挤到地板边缘越界
  - 小车自身冲出地板边界
  - 依赖“持续推送到最后一刻”，与“低速停放”条件天然冲突
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- role_id: crate_progress_toward_dock
  purpose: 奖励货箱向 dock 靠近的净进展（而非“当前离 dock 多近”的状态值），为主学习信号。
  why_required: 主目标是货箱到达 dock；没有主进度的稠密信号，仅靠稀疏成功事件无法学习。使用 delta 而非 proximity 是为了抑制“停在 dock 附近收割正分”的悬停陷阱。
  usable_signals: obs[12], obs[13]（货箱到 dock 偏移 ⇒ 距离）；可用 next_obs 与 obs 做差分。
  risks: 若不加门控，agent 可通过小幅来回振荡反复获得正 delta；收益与“净位移”不一致时可能诱发抖动。需与“净位置推进/终端事件”配合，并在速度异常高或损坏风险状态下降权。
  candidate operators: delta_state_signal（distance 差分为核心）、bounded_signal（对单步增益设上限）。

- role_id: docking_settle_success
  purpose: 在货箱真正进入 dock 并低速且朝向对齐时给出稀疏正向事件信号。
  why_required: 主目标本质上是一个稀疏的多条件状态事件，纯稠密信号无法表达“完成”这一里程碑；且低速沉降条件无法由距离 delta 表达。
  usable_signals: obs[12], obs[13]（距离小）、obs[8], obs[9]（货箱速度→近静止）、obs[10], obs[11]（货箱朝向）。推导方式：distance 阈值 + crate_speed 阈值 + 朝向与假定轴向一致的组合（derived_possible）。
  risks: dock 尺寸/朝向不可精确观测，阈值只能是近似；若阈值过松会误发（货箱擦过 dock），过严则信号极稀疏。
  candidate operators: sparse_terminal_bonus / event_bonus（带条件门）、condition_gated_bonus。

- role_id: fragile_handling_guard
  purpose: 抑制高速接触/强烈推挤，降低触发 3 次硬碰撞失败的频率。
  why_required: 失真是硬失败条件（3 次即终结），而货箱又不能被抓稳，唯一可控的“轻拿轻放”手段就是控制接触时的相对接近速度。
  usable_signals: obs[14]（接触）、
