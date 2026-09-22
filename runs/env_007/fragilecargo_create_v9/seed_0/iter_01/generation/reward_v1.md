# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- module-level episode state (no self, no extra helper, no info) ----
    state = getattr(compute_reward, "_v1_state", None)
    if state is None:
        state = {
            "prev_cc": None,      # previous cart->crate distance (m)
            "prev_cd": None,      # previous crate->dock distance (m)
            "prev_t": None,       # previous time_fraction, used to detect a fresh episode
            "entered": False,     # has crate ever fully entered dock tolerance
            "stable": 0,          # consecutive settled steps inside dock
            "hard": 0,            # accumulated hard-hit proxy count
            "success_paid": False,
            "fail_paid": False,
        }
        compute_reward._v1_state = state

    # time_fraction is monotone inside an episode; a drop means a new episode started
    t_now = float(obs[18])
    if state["prev_t"] is not None and t_now < state["prev_t"] - 0.05:
        state["prev_cc"] = None
        state["prev_cd"] = None
        state["entered"] = False
        state["stable"] = 0
        state["hard"] = 0
        state["success_paid"] = False
        state["fail_paid"] = False
    state["prev_t"] = t_now

    heading_x = float(obs[2])
    heading_y = float(obs[3])

    # ---- cart -> crate body-frame relative vector (metres) ----
    rel_x = float(obs[6]) * 3.0
    rel_y = float(obs[7]) * 3.0
    cc_now = (rel_x * rel_x + rel_y * rel_y) ** 0.5

    nrel_x = float(next_obs[6]) * 3.0
    nrel_y = float(next_obs[7]) * 3.0
    cc_next = (nrel_x * nrel_x + nrel_y * nrel_y) ** 0.5

    # ---- crate -> dock offset (metres) ----
    dock_x = float(next_obs[12]) * 5.0
    dock_y = float(next_obs[13]) * 4.0
    cd_next = (dock_x * dock_x + dock_y * dock_y) ** 0.5

    # ---- signed potential-difference signals ----
    if state["prev_cc"] is None:
        state["prev_cc"] = cc_next
    approach_cargo = 1.0 * (state["prev_cc"] - cc_next)
    state["prev_cc"] = cc_next

    if state["prev_cd"] is None:
        state["prev_cd"] = cd_next
    progress = 1.0 * (state["prev_cd"] - cd_next)
    state["prev_cd"] = cd_next

    # ---- one-shot bonus on first full dock containment, inferred from obs only ----
    inside = (abs(float(next_obs[12])) <= 0.024) and (abs(float(next_obs[13])) <= 0.030)
    dock_enter = 0.0
    if inside and not state["entered"]:
        dock_enter = 5.0
        state["entered"] = True

    # ---- roughness proxy from contact x closing speed (no impulse channel in obs) ----
    cart_speed = float(obs[4]) * 3.0
    crate_along = float(obs[8]) * 3.0 * heading_x + float(obs[9]) * 3.0 * heading_y
    closing = cart_speed - crate_along
    contact = float(obs[14]) > 0.5

    roughness = 0.0
    hard_hit = 0.0
    if contact and closing > 0.0:
        roughness = -0.02 * closing            # deliberately much weaker than progress
        if closing > 1.5:                      # high-severity contact proxy
            hard_hit = -0.5
            state["hard"] += 1

    # ---- effort / time bookkeeping ----
    action_cost = -0.0005 * (float(action[0]) ** 2 + float(action[1]) ** 2)
    time_cost = -0.002

    # ---- derived settling chain feeding terminal_success (no info flag available) ----
    crate_speed = ((float(next_obs[8]) * 3.0) ** 2 + (float(next_obs[9]) * 3.0) ** 2) ** 0.5
    aligned = float(next_obs[10]) > 0.866       # |heading error| < 30 deg via cos proxy
    if inside and aligned and crate_speed < 0.05:
        state["stable"] += 1
    else:
        state["stable"] = 0

    terminal_success = 0.0
    if state["stable"] >= 10 and not state["success_paid"]:
        terminal_success = 300.0
        state["success_paid"] = True

    # ---- derived failure chain feeding terminal_failure ----
    crate_wx = float(obs[0]) * 5.0 + rel_x * heading_x - rel_y * heading_y
    crate_wy = float(obs[1]) * 4.0 + rel_x * heading_y + rel_y * heading_x
    out_of_bounds = (
        abs(float(obs[0])) > 1.05
        or abs(float(obs[1])) > 1.05
        or abs(crate_wx) > 5.25
        or abs(crate_wy) > 4.2
    )

    terminal_failure = 0.0
    if (out_of_bounds or state["hard"] >= 3) and not state["fail_paid"]:
        terminal_failure = -100.0
        state["fail_paid"] = True

    components = {
        "approach_cargo": float(approach_cargo),
        "progress": float(progress),
        "dock_enter": float(dock_enter),
        "roughness": float(roughness),
        "action_cost": float(action_cost),
        "time_cost": float(time_cost),
        "hard_hit": float(hard_hit),
        "terminal_success": float(terminal_success),
        "terminal_failure": float(terminal_failure),
    }
    total_reward = (
        approach_cargo
        + progress
        + dock_enter
        + roughness
        + action_cost
        + time_cost
        + hard_hit
        + terminal_success
        + terminal_failure
    )
    return float(total_reward), components
```

# reward_v1 设计说明

**修复内容（仅合规修复，未改动奖励设计）**
上一版唯一触发 validation 的问题是禁止模式 `try:`：注释串 `dock en**try:**` 被正则命中。本次将相关注释改写为不含 `try:` 子串的措辞（`first full dock containment`、`crate -> dock offset` 等）。组件集合、权重数值、信号选择、模块级状态逻辑、一次性标志与重置判据与上一版逐字一致，奖励语义完全不变。全文件已复查：无 `import` / `class` / `try` / `except` / `eval` / `exec` / `open` / `self`，无未声明 info 字段，无未声明 obs 切片，未使用 `original_reward` 与 `training_progress`。

**selected task_family / dynamics_subtype**
`manipulation_grasping` / `staged_manipulation`：连续控制的非 prehensile 推箱任务，成功判据落在货箱位姿 + 朝向 + 低速 + 持续稳定。

**selected reward roles**

| role_id | 类别 | v1 中的实现 |
|---|---|---|
| `crate_delivery_progress` | mandatory | `progress`（货箱→坞距离本帧减少）+ `dock_enter`（首次完全入坞一次性奖励） |
| `gentle_contact_impact_control` | mandatory | `roughness`（接触 × 接近速度代理）+ `hard_hit`（高烈度接触事件惩罚） |
| `dock_settling_and_alignment` | mandatory | `terminal_success` 的派生稳定链（入坞 + 对齐 + 低速，连续 10 步） |
| `out_of_bounds_avoidance` | mandatory | `terminal_failure` 的越界分支 |
| `approach_cargo` | 结构块要求 | 小车→货箱距离本帧减少，作为主信号引导项 |
| `time_efficiency` | conditional | 恒定 `time_cost = -0.002/步`（结构块给定） |
| `action_smoothness` | conditional | 极小 `action_cost = -0.0005 * Σa²`，不足以压制推箱 |

**role_to_signal_mapping（实际用到的信号）**
- `progress` / `dock_enter` / settling：`obs[12], obs[13], next_obs[12], next_obs[13], next_obs[8], next_obs[9], next_obs[10]`
- `approach_cargo`：`obs[6], obs[7], next_obs[6], next_obs[7]`
- `roughness` / `hard_hit`：`obs[14], obs[4], obs[8], obs[9], obs[2], obs[3]`
- `out_of_bounds`：`obs[0], obs[1]` + 由 `obs[2], obs[3], obs[6], obs[7]` 还原的货箱世界坐标
- 未使用：`obs[5], obs[15], obs[16], obs[17]`（留待后续迭代）

**formula operator**
- `approach_cargo` / `progress`：`improvement_delta`（有符号、对称，靠近给正、远离给负），不用 proximity-only。
- `dock_enter`：`sparse_event_bonus` + 一次性标志。
- `roughness`：`bounded_signal` 线性截断 `-0.02 * max(0, closing) * contact`，量级远小于 `progress`。
- `hard_hit`：`indicator`（closing > 1.5 且接触）→ −0.5。
- `terminal_success`：`joint_condition_proxy` 连续计数版（`inside × aligned × slow` 累加满 10 步一次）。
- `terminal_failure`：`indicator`（越界 ∨ 累计硬冲击 ≥ 3）+ 一次性标志。
- `action_cost` / `time_cost`：`quadratic_penalty` 与恒定步成本，权重照结构块。

**excluded roles 及原因**
- `cart_forward_velocity_as_main`：小车奔跑与货箱交付无因果关系。
- `action_energy_penalty`：`action_energy` 为 forbidden info。
- `contact_impulse_direct_penalty`：`contact_impulse`、`hard_collision_count` 不可读。
- `proximity_only_dock_reward`：会诱发坞旁悬停刷分。
- `info_based_termination_penalty`：相关 info 字段全部禁用。
- `static_obstacle_avoidance` 强形式：v1 不加静态障碍 hinge，避免阻碍通行。

**为什么仍使用 terminal_success / terminal_failure**
卡片 `explicit_success_flag_available = false`、`explicit_failure_flag_available = false`，因此二者完全从 `obs`/`next_obs` 派生（入坞/越界判据见代码），并以一次性标志保证至多触发一次，且检测到 `obs[18]` 下降时整体重置。这是「已知奖励结构」块的强制要求，与卡片通用建议冲突时以结构块为准。

**留到后续迭代的职责**：静态障碍 hinge、显式时间效率加权、转向振荡抑制、更精确碰撞代理、边界前兆软惩罚。

**训练后应观察的 failure modes**
1. `roughness/hard_hit` 高频 → 高速撞击推箱，`hard` 常达 3；干预：改为 gate 乘到 `progress`。
2. `progress` 步均值≈0 且 `obs[18]` 走高 → 卡在隔墙开口/原地打转；干预：静态障碍 hinge。
3. 距离降到 ~0.2 m 后停滞 → delta 饱和，需坞附近低速/对齐 shaping。
4. `terminal_success` 从未触发但 `inside` 频繁真 → settling 门槛过难。
5. `terminal_failure` 主要来自货箱越界 → 核对体坐标系旋转符号。
6. 小车出界处反复触发 → 需边界前兆软信号。