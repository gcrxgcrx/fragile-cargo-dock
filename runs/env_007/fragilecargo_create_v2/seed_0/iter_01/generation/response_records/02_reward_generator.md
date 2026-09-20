# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    # 货箱到坞的有符号偏移（归一化）
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    # 货箱到坞距离（归一化尺度）
    dist = (dx * dx + dy * dy) ** 0.5
    next_dist = (ndx * ndx + ndy * ndy) ** 0.5

    # 货箱速率（世界系，m/s）
    crate_speed = ((obs[8] * 3.0) ** 2 + (obs[9] * 3.0) ** 2) ** 0.5

    # 货箱朝向误差（弧度），坞假设与坐标轴对齐
    crate_heading = (obs[11] / (1.0 + 1e-6)) if False else 0.0  # placeholder, replaced below
    # 用 atan2 计算朝向角，再取与 0 的夹角（坞对齐假设为 0 朝向）
    import_guard = 0.0  # no-op, keep function pure
    ch = obs[10]
    sh = obs[11]
    # 朝向角
    heading_angle = 0.0
    # 手动 atan2 近似：使用 cos/sin 直接构造对齐度，避免 atan2 依赖
    # 对齐度 = cos(heading)（坞朝向假设为 0，即 cos=1 表示对齐）
    align_cos = ch  # 范围 [-1, 1]，1 表示完全对齐

    # 接触标志
    contact = obs[14]

    # 小车前向速度（归一化）
    cart_v = obs[4]

    # ---------- 组件 A: 货箱向坞的进度（delta 形式，防悬停） ----------
    # 用 improvement_delta：距离减少为正
    progress = (dist - next_dist)  # 正=靠近
    # 低速门控：接近坞时抑制速度，防止高速滑过
    # 门控因子在 dist 小时衰减，但不阻断早期探索（dist 大时接近 1）
    # 使用线性衰减门：dist < 0.15 时开始衰减
    near_gate = 1.0
    if dist < 0.15:
        near_gate = max(0.2, dist / 0.15)  # 最低保留 0.2，避免完全阻断
    # 货箱速率门控：接近坞时若速度高，抑制进度奖励
    speed_gate = 1.0
    if dist < 0.15:
        # 速度越高，门控越低（但保留最低 0.2）
        speed_gate = max(0.2, 1.0 / (1.0 + 2.0 * crate_speed))
    progress_reward = 5.0 * progress * near_gate * speed_gate

    # ---------- 组件 B: 货箱朝向对齐（仅在接近坞时启用） ----------
    # 对齐度：align_cos 在 [-1,1]，映射到 [0,1]
    align_factor = (align_cos + 1.0) * 0.5  # 0=反向, 1=对齐
    # 仅在货箱接近坞时启用（dist < 0.3）
    align_gate = 0.0
    if dist < 0.3:
        align_gate = max(0.0, 1.0 - dist / 0.3)
    alignment_reward = 0.5 * align_factor * align_gate

    # ---------- 组件 C: 货箱在坞内近静止（settling，仅在近坞时启用） ----------
    # 速率惩罚：仅在货箱接近坞时启用，避免抑制必要推动
    settle_gate = 0.0
    if dist < 0.2:
        settle_gate = max(0.0, 1.0 - dist / 0.2)
    # 速率超过 0.05 m/s 时惩罚（hinge 形式）
    speed_excess = max(0.0, crate_speed - 0.05)
    settling_penalty = -1.0 * speed_excess * settle_gate

    # ---------- 组件 D: 边界/障碍安全（hinge 形式，轻量） ----------
    # 小车位置越界风险：obs[0], obs[1] 接近 ±1 时惩罚
    cart_x = obs[0]
    cart_y = obs[1]
    boundary_penalty = 0.0
    # 小车边界 hinge：|x| > 0.85 或 |y| > 0.85 时惩罚
    if abs(cart_x) > 0.85:
        boundary_penalty -= 0.5 * (abs(cart_x) - 0.85)
    if abs(cart_y) > 0.85:
        boundary_penalty -= 0.5 * (abs(cart_y) - 0.85)
    # 障碍接近度惩罚：sensor 接近 1 时轻罚
    sensor_max = max(obs[15], obs[16], obs[17])
    if sensor_max > 0.8:
        boundary_penalty -= 0.3 * (sensor_max - 0.8)

    # ---------- 组件 E: 接触冲击抑制（保守，仅在接触时） ----------
    # 接触时若货箱速度高，轻罚（间接推断硬碰撞风险）
    impact_penalty = 0.0
    if contact > 0.5:
        # 接触时货箱速率超过 1.0 m/s 时轻罚
        impact_excess = max(0.0, crate_speed - 1.0)
        impact_penalty = -0.3 * impact_excess

    # ---------- 汇总 ----------
    total_reward = (
        progress_reward
        + alignment_reward
        + settling_penalty
        + boundary_penalty
        + impact_penalty
    )

    components = {
        "crate_to_dock_progress": float(progress_reward),
        "crate_dock_alignment": float(alignment_reward),
        "crate_settling": float(settling_penalty),
        "boundary_avoidance": float(boundary_penalty),
        "soft_contact_penalty": float(impact_penalty),
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- task_family: `manipulation_grasping`
- dynamics_subtype: `goal_approach_and_soft_contact`
- control_type: continuous（2 维：drive + steer）

## selected reward roles
| role_id | 类型 | 是否纳入 v1 | 理由 |
|---|---|---|---|
| crate_to_dock_progress | mandatory | ✅ | 主学习信号，delta 形式防悬停 |
| crate_dock_alignment | mandatory | ✅ | 成功条件要求朝向 < 30°，近坞时启用 |
| crate_settling | mandatory | ✅ | 成功条件要求速度 < 0.05 m/s，近坞时启用 |
| soft_contact_penalty | conditional | ✅（保守） | 用接触 + 货箱速率间接推断冲击风险 |
| boundary_avoidance | conditional | ✅ | 用小车位置 + 障碍传感器 hinge 防越界 |
| time_efficiency | conditional | ❌ | 任务未强调速度，且可能诱导冒险推快 |
| action_smoothness | conditional | ❌ | 无 previous action，且属附属优化 |

## role_to_signal_mapping
| role_id | 使用信号 | formula operator |
|---|---|---|
| crate_to_dock_progress | obs[12], obs[13], next_obs[12], next_obs[13] | improvement_delta + 低速门控（soft_health_gate 思想） |
| crate_dock_alignment | obs[10], obs[11] | bounded_signal（cos 映射到 [0,1]）+ 近坞门控 |
| crate_settling | obs[8], obs[9] | hinge_penalty（速率超过 0.05 才罚）+ 近坞门控 |
| soft_contact_penalty | obs[14], obs[8], obs[9] | hinge_penalty（接触时速率超 1.0 才罚） |
| boundary_avoidance | obs[0], obs[1], obs[15..17] | hinge_penalty（位置/传感器超阈值才罚） |

## excluded roles 及原因
- `hard_collision_penalty_direct`：`hard_collision_count`、`contact_impulse`、`termination_reason` 均 forbidden，无法可靠计算。
- `success_bonus_from_info`：`is_success`、`cargo_inside_dock`、`stable_steps` 均 forbidden。
- `official_reward_shaping`：`original_reward` 被 mask。
- `cart_forward_velocity_as_main`：小车速度不是任务目标，货箱才是被操控对象。
- `energy_penalty_default`：任务未要求节能。
- `time_efficiency`：任务未强调速度，且可能诱导冒险推快增加硬碰撞风险。
- `action_smoothness`：无 previous action 历史，且属附属优化。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- `explicit_success_flag_available: false`，`is_success` 在 forbidden 列表中。
- `explicit_failure_flag_available: false`，`termination_reason`、`hard_collision_count` 均 forbidden。
- `allowed_info_fields: []`，无任何允许读取的 info 字段。
- 因此 v1 完全依赖 obs 中的连续信号构造稠密奖励，不依赖任何终止事件标志。

## 留到后续迭代的职责
- `time_efficiency`：若观察到时间耗尽截断频繁，可加入 obs[18] 相关 shaping。
- `action_smoothness`：若观察到动作抖动导致货箱失控，可加入 action 二次惩罚。
- 更精细的硬碰撞推断：若 obs[14] + 速度突变能可靠识别硬碰撞，可升级 soft_contact_penalty。
- 动态课程：若早期探索与后期精细控制冲突明显，可引入 training_progress 加权。

## 关键设计决策

### 1. 主信号使用 delta 而非 proximity
`progress = dist - next_dist`，占据某个状态本身不得分，只有"更接近坞"才得分。这直接防止"货箱停在坞附近但不入坞"的悬停陷阱。

### 2. 低速门控防止高速滑过
当 `dist < 0.15` 时，`near_gate` 和 `speed_gate` 同时衰减进度奖励。货箱速度越高，`speed_gate` 越低，抑制"高速冲过坞"的行为。但最低保留 0.2，不阻断早期探索（远离坞时门控为 1）。

### 3. settling 仅在近坞时启用
`settle_gate` 在 `dist < 0.2` 时才非零，避免在货箱远离坞时惩罚速度（那会抑制必要的推动）。

### 4. alignment 仅在近坞时启用
`align_gate` 在 `dist < 0.3` 时才非零，避免货箱远离坞时朝向误差无意义地影响奖励。

### 5. 辅助组件不与任务推进对抗（自检）

**状态 ①：智能体什么都不做，货箱静止在初始位置**
- 假设初始 `dist = 0.5`（远离坞），`next_dist = 0.5`（静止）
- `progress = 0`，`near_gate = 1`，`speed_gate = 1` → `progress_reward = 0`
- `align_gate = 0`（dist > 0.3）→ `alignment_reward = 0`
- `settle_gate = 0`（dist > 0.2）→ `settling_penalty = 0`
- `boundary_penalty = 0`（假设小车在中心）
- `impact_penalty = 0`（无接触）
- **总分 = 0.0**

**状态 ②：智能体正在把货箱推向坞（货箱有速度）**
- 假设 `dist = 0.5`，`next_dist = 0.45`（靠近 0.05），`crate_speed = 0.5 m/s`
- `progress = 0.05`，`near_gate = 1`（dist > 0.15），`speed_gate = 1` → `progress_reward = 5.0 * 0.05 = 0.25`
- `align_gate = 0`（dist > 0.3）→ `alignment_reward = 0`
- `settle_gate = 0`（dist > 0.2）→ `settling_penalty = 0`
- `boundary_penalty = 0`
- `impact_penalty = 0`（假设无接触或速度 < 1.0）
- **总分 = 0.25**

**状态 ② (0.25) > 状态 ① (0.0)** ✅ 自检通过。

### 6. 组件量级平衡
- `progress_reward`：典型值 ±0.25（每步距离变化 ~0.05 × 5.0）
- `alignment_reward`：最大 0.5（近坞且完全对齐）
- `settling_penalty`：典型值 -0.1 ~ -0.5（近坞且速度高时）
- `boundary_penalty`：典型值 0 ~ -0.5（接近边界时）
- `impact_penalty`：典型值 0 ~ -0.3（接触且速度高时）

主信号 `progress_reward` 在远离坞时主导，辅助组件在近坞/危险时生效，量级可比。

## failure mode 覆盖表

| failure_mode | 对应组件 | 防止机制 |
|---|---|---|
| 货箱停在坞附近但不入坞（悬停） | crate_to_dock_progress | 使用 delta(distance) 而非 proximity，静止时 progress=0，只有靠近才得分 |
| 货箱高速滑过坞 | crate_to_dock_progress (speed_gate) + crate_settling | 近坞时 speed_gate 衰减进度奖励；settling_penalty 对高速 hinge 惩罚 |
| 硬碰撞累积导致失败 | soft_contact_penalty | 接触时若货箱速率 > 1.0 m/s 轻罚，间接抑制硬冲击 |
| 货箱或小车越界 | boundary_avoidance | 小车位置 |x|>0.85 或 |y|>0.85 时 hinge 惩罚；障碍传感器 > 0.8 时轻罚 |
| 撞隔墙/卡在开口 | boundary_avoidance | 障碍接近度传感器 obs[15..17] > 0.8 时 hinge 惩罚，引导远离障碍 |
| 动作抖动导致货箱失控 | 未覆盖（留到后续迭代） | 无 previous action 历史，v1 不加入 action_smoothness；若观察到抖动问题，后续迭代加入 |
| 时间耗尽截断 | 未覆盖（留到后续迭代） | 任务未强调速度，v1 不加入 time_efficiency；若截断频繁，后续迭代加入 obs[18] shaping |
| 主信号被悬停收割 | crate_to_dock_progress | 使用 delta 形式，静止状态得分为 0，无法通过悬停刷分 |
