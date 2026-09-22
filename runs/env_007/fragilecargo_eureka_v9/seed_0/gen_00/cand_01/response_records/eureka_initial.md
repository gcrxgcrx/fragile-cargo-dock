# Response Record

```python
# ---------- 模块级状态（一次性事件 / 回合边界检测 / 失败守卫） ----------
_PREV_T = [-1.0]        # 上一帧 obs[18]（已用时间比例，episode 内单调递增，重置回落）
_STREAK = [0]           # “泊位内 + 对齐 + 慢”的连续计数
_PAID = [False]         # terminal_success 是否已发放（整局只发一次）
_ENTERED = [False]      # 货箱是否曾经完全进入泊位容差（dock_enter 只发一次）
_HARD_HITS = [0]        # 观测代理累计的硬冲击次数
_FAIL_PAID = [False]    # terminal_failure 是否已发放

# ===== 自检记录（量级估算，单位：单步奖励） =====
# 自检①: idle（小车不动、货箱静止在初始位置）        R_idle    ≈ -0.002
#         正在把货箱推向泊位（正常推箱，接触但 closing≈0） R_push    ≈ +0.03 (0.02~0.05)
#         => R_push > R_idle，严格成立。
# 自检②: 货箱停在泊位外 0.3m、速度≈0、保持 400 步      ≈ -0.8（每步只有 -0.002 时间成本，无任何状态正项）
#         真正进入容差并停稳、连续 10 步后 episode 终止 ≈ +30
#         （settle_dwell 0.5×10 + dock_enter 5 + terminal_success 300 被单步裁剪到 20）
#         => ④ 严格高于 ③。
# 自检③: ⑤接触 + closing=1.0m/s : roughness -0.1 , hard_hit -0.5 , 推进 +0.03  => ≈ -0.57
#         ⑥接触 + closing=0.05m/s: roughness -0.005, 推进 +0.03              => ≈ +0.025
#         差距 ≈ 0.60，约为正常推箱单步推进量(0.03)的 20 倍 => 轻柔度足够强。
#         接触但未接近(closing≈0)时 roughness=0、hard_hit=0 => 正常匀速推箱不被判为撞击。
# 自检④: 在同一“停稳”状态连续调用 12 次，每步增量恒为 0.5-0.002 = +0.498（线性增长，
#         不因 _PAID / _STREAK 达标而被关闭）。
# 自检⑤: R_idle ≈ -0.002 < R_push ≈ +0.03 < R_settled ≈ +0.50
#         R_push - R_idle ≈ 0.03 > 常驻罚项单步上界(|action_cost|+|time_cost| ≈ 0.003)。
#         边界惩罚只在 |pos|>0.9 时生效；撞击类惩罚只在 closing>0 且接触时生效，
#         均不在正常推进路径上常驻，故不会把“行动”变成比“不动”更差。


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 0) 回合边界检测 + 模块级状态重置（obs[18] 单调递增，重置时回落） ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] - 1e-9 or t <= (1.0 / 400.0):
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
        _FAIL_PAID[0] = False
    _PREV_T[0] = t

    ax = 5.0   # 仓库半宽（obs[0] / obs[12] 的归一化分母）
    ay = 4.0   # 仓库半高（obs[1] / obs[13] 的归一化分母）

    # ---------- 1) approach_cargo：小车→货箱距离本帧缩短量（米，有符号、对称） ----------
    d_cart_crate = 3.0 * (obs[6] * obs[6] + obs[7] * obs[7]) ** 0.5
    n_d_cart_crate = 3.0 * (next_obs[6] * next_obs[6] + next_obs[7] * next_obs[7]) ** 0.5
    approach_cargo = 1.0 * (d_cart_crate - n_d_cart_crate)

    # ---------- 2) progress：货箱→泊位中心距离本帧缩短量（米，有符号、对称） ----------
    odx = obs[12] * ax
    ody = obs[13] * ay
    d_crate_dock = (odx * odx + ody * ody) ** 0.5
    ndx = next_obs[12] * ax
    ndy = next_obs[13] * ay
    n_d_crate_dock = (ndx * ndx + ndy * ndy) ** 0.5
    progress = 1.0 * (d_crate_dock - n_d_crate_dock)

    # ---------- 3) 货箱世界坐标（车体系相对位置旋转到世界系后加小车位置） ----------
    n_cos_h = next_obs[2]
    n_sin_h = next_obs[3]
    n_rx = next_obs[6] * 3.0
    n_ry = next_obs[7] * 3.0
    n_crate_x = next_obs[0] * ax + n_rx * n_cos_h - n_ry * n_sin_h
    n_crate_y = next_obs[1] * ay + n_rx * n_sin_h + n_ry * n_cos_h

    # ---------- 4) 完成谓词（显式从 obs 推断）：泊位容差内 + 朝向对齐 + 速度接近 0 ----------
    in_tolerance = 1.0 if (abs(ndx) < 0.25 and abs(ndy) < 0.25) else 0.0
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    well_slowed = 1.0 if crate_speed < 0.05 else 0.0
    aligned = 1.0 if abs(next_obs[10]) >= 0.866 else 0.0   # 朝向误差 < 30 度
    settled = 1.0 if (in_tolerance > 0.5 and well_slowed > 0.5 and aligned > 0.5) else 0.0

    # ---------- 5) dock_enter：货箱首次完全进入泊位容差时的一次性奖励 ----------
    dock_enter = 0.0
    if in_tolerance > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # ---------- 6) 停稳期每步收益（谓词成立就每步发放，不被任何其它开关关闭） ----------
    settle_dwell = 0.5 * settled

    # ---------- 7) terminal_success：连续 10 步满足完成谓词时一次性发放 ----------
    if settled > 0.5:
        _STREAK[0] = _STREAK[0] + 1
    else:
        _STREAK[0] = 0
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0

    # ---------- 8) roughness：接触冲量比例惩罚的可观测代理（接近速度 × 接触） ----------
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    # 只在“接触中且正在接近”时惩罚；匀速推箱 closing≈0，不罚（不会压制必要推进）
    roughness = -0.1 * contact * closing

    # ---------- 9) hard_hit：单步内一次高速冲击的固定惩罚（冲量不可读，用高接近速度代理） ----------
    hard_hit = 0.0
    if contact > 0.5 and closing > 0.8:
        hard_hit = -0.5
        _HARD_HITS[0] = _HARD_HITS[0] + 1

    # ---------- 10) action_cost ----------
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])

    # ---------- 11) time_cost ----------
    time_cost = -0.002

    # ---------- 12) bounds_safety：小车/货箱接近场地边缘时单调增强的惩罚 ----------
    b_cart_x = abs(next_obs[0]) - 0.9
    if b_cart_x < 0.0:
        b_cart_x = 0.0
    b_cart_y = abs(next_obs[1]) - 0.9
    if b_cart_y < 0.0:
        b_cart_y = 0.0
    b_crate_x = abs(n_crate_x) / ax - 0.9
    if b_crate_x < 0.0:
        b_crate_x = 0.0
    b_crate_y = abs(n_crate_y) / ay - 0.9
    if b_crate_y < 0.0:
        b_crate_y = 0.0
    bounds_safety = -40.0 * (
        (b_cart_x / 0.15) ** 2
        + (b_cart_y / 0.15) ** 2
        + (b_crate_x / 0.15) ** 2
        + (b_crate_y / 0.15) ** 2
    )

    # ---------- 13) terminal_failure：越界 / 累计硬冲击 >= 3 的一次性惩罚 ----------
    cart_out = 1.0 if (abs(next_obs[0]) > 1.05 or abs(next_obs[1]) > 1.05) else 0.0
    crate_out = 1.0 if (abs(n_crate_x) > ax * 1.05 or abs(n_crate_y) > ay * 1.05) else 0.0
    terminal_failure = 0.0
    if (cart_out > 0.5 or crate_out > 0.5 or _HARD_HITS[0] >= 3) and not _FAIL_PAID[0]:
        _FAIL_PAID[0] = True
        terminal_failure = -100.0

    total_reward = (
        approach_cargo
        + progress
        + dock_enter
        + settle_dwell
        + roughness
        + action_cost
        + time_cost
        + hard_hit
        + bounds_safety
        + terminal_success
        + terminal_failure
    )

    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "dock_enter": dock_enter,
        "settle_dwell": settle_dwell,
        "roughness": roughness,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "hard_hit": hard_hit,
        "bounds_safety": bounds_safety,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
    }

    return (float(total_reward), components)
```
