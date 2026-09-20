# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 恢复关键量 ----
    # 货箱到泊位偏移（归一化 -> 米）
    dx_cur = obs[12] * 5.0
    dy_cur = obs[13] * 4.0
    dx_next = next_obs[12] * 5.0
    dy_next = next_obs[13] * 4.0

    dist_cur = (dx_cur * dx_cur + dy_cur * dy_cur) ** 0.5
    dist_next = (dx_next * dx_next + dy_next * dy_next) ** 0.5

    # 货箱速度（世界系，归一化 -> m/s）
    cvx = obs[8] * 3.0
    cvy = obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 小车前向速度
    cart_fwd_speed = obs[4] * 3.0

    # 接触标志
    contact = obs[14]

    # ---- 组件 A: 货箱向泊位推进（增量信号，避免悬停刷分）----
    # 用 improvement_delta：只有"更接近"才得分，占据某状态本身不得分
    progress_delta = dist_cur - dist_next  # 正=靠近
    # 接近程度因子：越接近泊位，越抑制推进速度（防止冲过头）
    # 远离时因子接近 1（不阻断探索），接近时衰减
    near_factor = 1.0 / (1.0 + 0.6 * max(0.0, 2.0 - dist_cur))
    # 只在接触推箱时给推进奖励（避免小车空跑刷分）
    push_gate = 1.0 if contact > 0.5 else 0.0
    progress_reward = 3.0 * progress_delta * near_factor * push_gate

    # ---- 组件 B: 停靠对齐与低速（条件化，仅在接近泊位时激活）----
    # 接近程度门：dist < 1.5m 时开始生效，平滑过渡
    dock_proximity = max(0.0, min(1.0, (1.5 - dist_cur) / 1.5))
    # 低速因子：货箱速度越低越好（bounded，0~1）
    speed_factor = 1.0 / (1.0 + 4.0 * crate_speed)
    # 朝向对齐因子：货箱朝向与泊位朝向对齐（泊位朝向近似为 0，即 cos 接近 1）
    # 用 |cos| 的平滑形式，避免 atan2 分支
    crate_cos = obs[10]
    align_factor = max(0.0, crate_cos)  # 0~1，对齐时接近 1
    # 联合条件 proxy（几何平均，避免塌缩）
    settle_proxy = (speed_factor * align_factor) ** 0.5
    settle_reward = 1.5 * dock_proximity * settle_proxy

    # ---- 组件 C: 易碎冲击抑制（hinge，仅在接触且高速时生效）----
    # 接触时小车前向速度超过阈值才惩罚，避免误罚正常推箱
    impact_excess = max(0.0, cart_fwd_speed - 1.2)
    impact_penalty = -0.8 * impact_excess * contact

    total_reward = progress_reward + settle_reward + impact_penalty

    components = {
        "crate_to_dock_progress": float(progress_reward),
        "crate_settle_and_align": float(settle_reward),
        "fragile_impact_avoidance": float(impact_penalty),
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- task_family: `manipulation_grasping`
- dynamics_subtype: `staged_manipulation`
- control_type: `continuous`（drive + steer）

## selected reward roles
| role_id | 类型 | 是否纳入 v1 |
|---|---|---|
| crate_to_dock_progress | mandatory | ✅ 组件 A |
| crate_settle_and_align | mandatory | ✅ 组件 B |
| fragile_impact_avoidance | conditional | ✅ 组件 C |
| boundary_avoidance | conditional | ❌ 留后续 |
| obstacle_avoidance | conditional | ❌ 留后续 |
| action_smoothness_or_energy | conditional | ❌ 留后续 |
| official_reward_shaping / info_based_success_bonus / hard_collision_penalty_from_info / push_until_end / pure_proximity_hover | avoid | ❌ 禁用 |

## role_to_signal_mapping
| role | signal | formula operator |
|---|---|---|
| crate_to_dock_progress | obs[12], obs[13]（当前/下一步偏移） | `improvement_delta` + `bounded_signal`（near_factor 门控） |
| crate_settle_and_align | obs[8], obs[9], obs[10], obs[12], obs[13] | `joint_condition_proxy`（几何平均）+ `bounded_signal` |
| fragile_impact_avoidance | obs[14], obs[4] | `dense_state_signal`（hinge） |

## 每个 role 的 formula operator 选择理由
- **A（推进）**：用 `improvement_delta`（`dist_cur - dist_next`）而非 proximity，避免 `pure_proximity_hover`。乘 `near_factor = 1/(1+0.6*max(0,2-dist))` 在接近泊位时衰减推进奖励，防止冲过头；远离时因子≈1，不阻断探索。乘 `push_gate`（接触才给分）避免小车空跑刷分。
- **B（停靠）**：用 `joint_condition_proxy` 几何平均 `(speed_factor * align_factor)**0.5`，避免裸乘积塌缩。乘 `dock_proximity` 门控，只在接近泊位时激活，不抑制接近阶段。
- **C（冲击）**：用 hinge `max(0, cart_fwd_speed - 1.2) * contact`，只在接触且速度超阈值时惩罚，避免误罚正常推箱。

## excluded roles 及原因
- **boundary_avoidance**：边界几何未显式给出，阈值需推断，v1 暂不引入以免误罚。
- **obstacle_avoidance**：开口几何未显式给出，可能误罚穿开口行为，留后续迭代。
- **action_smoothness_or_energy**：任务未强调效率，v1 默认不加。
- **official_reward_shaping / info_based_success_bonus / hard_collision_penalty_from_info**：信号被 forbidden，无法使用。
- **push_until_end**：与低速停靠条件冲突，禁用。
- **pure_proximity_hover**：会导致悬停刷分，禁用。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- `explicit_success_flag_available: false`，`explicit_failure_flag_available: false`，`allowed_info_fields: []`。
- 成功/失败只能通过观测间接推断，v1 不伪造 success/failure flag，改用组件 B 的连续 proxy 引导停靠。

## 留到后续迭代的职责
- 边界避让、障碍避让、动作平滑/能耗、时间调度（time_fraction 相关）。
- 若训练显示冲过头严重，可加强 near_factor 或加入接近时的速度 hinge 惩罚。

## failure mode 覆盖表

| failure_mode | 对应组件 | 防止机制 |
|---|---|---|
| 货箱被推过头冲出泊位 | crate_to_dock_progress | `near_factor` 在接近泊位时衰减推进奖励，抑制高速冲入 |
| 货箱进入泊位但速度不达标 | crate_settle_and_align | `speed_factor = 1/(1+4*speed)` 在接近时奖励低速 |
| 货箱朝向未对齐 | crate_settle_and_align | `align_factor = max(0, cos)` 奖励朝向对齐 |
| 硬冲击导致货箱损坏 | fragile_impact_avoidance | hinge 惩罚接触时小车前向速度 > 1.2 m/s |
| 小车或货箱出界 | 未覆盖 | 边界几何未显式给出，留后续迭代 |
| 卡在隔墙开口 | 未覆盖 | 开口几何未显式给出，留后续迭代 |
| 悬停在泊位附近不完成 | crate_to_dock_progress | 用 delta 而非 proximity，占据状态不得分 |
| 时间耗尽截断 | 未覆盖 | v1 未引入时间调度，留后续迭代 |

## 必做自检（激励冲突检查）

**状态 ①：小车什么都不做，货箱静止在初始位置**
- 假设货箱初始距泊位约 4.0 m，`dist_cur = dist_next = 4.0`，`progress_delta = 0`
- `contact = 0` → `push_gate = 0` → `progress_reward = 0`
- `dock_proximity = max(0, (1.5-4.0)/1.5) = 0` → `settle_reward = 0`
- `impact_penalty = 0`（无接触）
- **总分 = 0.0**

**状态 ②：小车正在把货箱推向泊位（货箱有速度）**
- 假设 `dist_cur = 3.0`，`dist_next = 2.9`，`progress_delta = 0.1`
- `near_factor = 1/(1+0.6*max(0,2-3)) = 1/(1+0) = 1.0`
- `contact = 1` → `push_gate = 1` → `progress_reward = 3.0 * 0.1 * 1.0 * 1.0 = 0.3`
- `dock_proximity = max(0, (1.5-3.0)/1.5) = 0` → `settle_reward = 0`
- 假设 `cart_fwd_speed = 1.0 < 1.2` → `impact_penalty = 0`
- **总分 = 0.3**

**结论：状态 ② (0.3) > 状态 ① (0.0)，通过自检。** 辅助组件（near_factor、settle、impact）均不会对"把货箱推向目标位置"产生净负贡献。