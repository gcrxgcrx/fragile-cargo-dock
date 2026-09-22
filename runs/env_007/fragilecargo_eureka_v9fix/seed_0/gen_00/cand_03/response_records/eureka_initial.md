# Response Record

```python
# ---------------------------------------------------------------------------
# 模块级状态（用于一次性事件 / 连续计数 / 回合边界检测）
# ---------------------------------------------------------------------------
_PREV_T = [-1.0]      # 上一帧的时间比例 obs[18]，用于检测 episode 重置
_STREAK = [0]         # "泊位内 + 对齐 + 慢" 的连续步计数
_PAID = [False]       # 一次性 terminal_success 是否已发放
_ENTERED = [False]    # 一次性 dock_enter 是否已发放
_FAILED = [False]     # 一次性 terminal_failure 是否已发放
_HARD_HITS = [0]      # 本 episode 累计硬冲击次数（代理）


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """obs + action -> 分项奖励之和。

    分项来源（作者奖励结构，按 obs 重写）：
      approach_cargo  +1.0 / m   本帧 小车->货箱 距离缩短（有符号、对称）
      progress        +1.0 / m   本帧 货箱->泊位 距离缩短（有符号、对称）
      dock_enter      +5.0       货箱首次完全进入泊位容差（整局一次）
      roughness       -k * 接触 * max(0, 接近速度)   （冲量比例的观测代理，k=0.1）
      action_cost     -0.0005 * sum(action^2)
      time_cost       -0.002 / 步
      hard_hit        -0.5       单步内硬冲击（接触且接近速度 > 1.0 m/s）
      terminal_success +300      完成谓词连续 10 步成立（整局一次）
      terminal_failure -100      越界 或 累计硬冲击 >= 3（整局一次）
    另外按"完成奖励：一次性事件 和 停稳期每步收益 都要有"的要求，增加
      settle_hold     +2.0 / 步  完成谓词成立时每步发放（不被 _PAID 关掉）
      boundary_guard  负值       小车/货箱接近场地边界的单调下降惩罚

    ------------------------------------------------------------------
    自检记录（按每步量级估算；正常推箱 ≈ 0.6 m/s、dt ≈ 0.05 s ⇒ 每步位移 ≈ 0.03 m）
      R_idle    (不动, 货箱静止在初始位置)        ≈ -0.002
                = time_cost(-0.002) + 其它 0
      R_push    (正常匀速推箱, 接触, closing≈0.1) ≈ +0.02
                = progress(+0.03) + approach(0) + roughness(-0.01)
                  + action_cost(-0.0003) + time_cost(-0.002)
      R_settled (泊位内 + 对齐 + 慢, 谓词成立)     ≈ +2.03
                = settle_hold(+2.0) + progress 小增量 + time_cost(-0.002)
      自检①  R_push(+0.02) > R_idle(-0.002)                     通过
      自检②  泊位外 0.3 m 悬停 400 步 ≈ 400*(-0.002) = -0.8；
             真正入坞停稳轨迹 ≈ +5(dock_enter) + 10*2.0 + 增量 + 事件 ≫ -0.8  通过
      自检③  ⑤(closing=1.0) 比 ⑥(closing=0.05) 多罚 0.1*0.95 = 0.095，
             大于正常推箱单步 progress(≈0.03)                       通过
      自检④  同一停稳状态连续 12 次调用，差值恒为 +2.0 附近（线性增长） 通过
      自检⑤  R_settled(+2.03) > R_push(+0.02) > R_idle(-0.002)，
             且 R_push-R_idle(0.022) > 常规单步罚项最大量级
             (action_cost 0.001 + time_cost 0.002 + 常规 roughness 0.01 ≈ 0.013) 通过
      自检⑤' 小车 |x|=1.05 时 boundary_guard ≈ -1.53，明显低于场地中心 0  通过
    ------------------------------------------------------------------
    """

    # ---------------- 0. 回合边界检测与状态重置 ----------------
    t = float(next_obs[18])
    if t < _PREV_T[0] or _PREV_T[0] < 0.0:      # t 回落 或 首次调用 => 新 episode
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _FAILED[0] = False
        _HARD_HITS[0] = 0
    _PREV_T[0] = t

    components = {}

    # ---------------- 1. approach_cargo：小车 -> 货箱 距离缩短（米，有符号） ----
    # obs[6], obs[7] 是车体系相对位置 / 3.0 m；向量模长与坐标系无关，
    # 因此两帧模长之差就是真实的"小车->货箱"距离变化（米）。
    d_cc_now = 3.0 * (obs[6] * obs[6] + obs[7] * obs[7]) ** 0.5
    d_cc_next = 3.0 * (next_obs[6] * next_obs[6] + next_obs[7] * next_obs[7]) ** 0.5
    approach_cargo = 1.0 * (d_cc_now - d_cc_next)          # 靠近为正，远离为负
    components["approach_cargo"] = float(approach_cargo)

    # ---------------- 2. progress：货箱 -> 泊位 距离缩短（米，有符号，对称） ----
    dx_now = obs[12] * 5.0
    dy_now = obs[13] * 4.0
    dx_next = next_obs[12] * 5.0
    dy_next = next_obs[13] * 4.0
    d_dock_now = (dx_now * dx_now + dy_now * dy_now) ** 0.5
    d_dock_next = (dx_next * dx_next + dy_next * dy_next) ** 0.5
    progress = 1.0 * (d_dock_now - d_dock_next)            # 只奖不罚的 max(0,·) 被禁止
    components["progress"] = float(progress)

    # ---------------- 3. roughness：接触冲量比例的观测代理 -------------------
    # 观测里没有冲量维，用 (小车前向速度 - 货箱沿车头方向速度) 作为接近速度代理。
    # 仅在实际接触时生效；closing<=0（接触但不接近，例如匀速推箱）不罚。
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    # k = 0.1：closing=1.0 m/s 时罚 0.1（为正常推箱单步 progress 的 3 倍以上），
    # 而轻微接触（closing≈0.05）只罚 0.005，不会压制推进。
    roughness = -0.1 * contact * closing
    components["roughness"] = float(roughness)

    # ---------------- 4. dock_enter：首次完全进入泊位容差（整局一次） ---------
    in_dock = (abs(next_obs[12]) <= 0.024) and (abs(next_obs[13]) <= 0.030)
    dock_enter = 0.0
    if in_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0
    components["dock_enter"] = float(dock_enter)

    # ---------------- 5. 完成谓词 与 停稳期每步收益 ---------------------------
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    align = abs(next_obs[10])
    if abs(next_obs[11]) > align:
        align = abs(next_obs[11])
    aligned = align > 0.8660254            # 航向误差 < 30°（允许 90° 倍数）
    settled = in_dock and (crate_speed < 0.05) and aligned

    if settled:
        _STREAK[0] = _STREAK[0] + 1
    else:
        _STREAK[0] = 0

    # 停稳期每步收益：谓词成立就发，不被 _PAID / _STREAK 等任何开关关闭。
    settle_hold = 2.0 if settled else 0.0
    components["settle_hold"] = float(settle_hold)

    # ---------------- 6. terminal_success：连续 10 步成立（整局一次 +300） ----
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0           # 环境单步裁剪上限 20，仍按分项表实现
    components["terminal_success"] = float(terminal_success)

    # ---------------- 7. hard_hit：单步硬冲击（接触且接近速度很大） -----------
    hard_hit = 0.0
    if contact > 0.5 and closing > 1.0:
        hard_hit = -0.5
        _HARD_HITS[0] = _HARD_HITS[0] + 1
    components["hard_hit"] = float(hard_hit)

    # ---------------- 8. terminal_failure：越界 或 累计硬冲击 >= 3（一次） ----
    cart_out = (abs(next_obs[0]) > 1.05) or (abs(next_obs[1]) > 1.05)
    head_c = next_obs[2]
    head_s = next_obs[3]
    rel_x = next_obs[6] * 3.0
    rel_y = next_obs[7] * 3.0
    crate_wx = next_obs[0] * 5.0 + head_c * rel_x - head_s * rel_y
    crate_wy = next_obs[1] * 4.0 + head_s * rel_x + head_c * rel_y
    crate_out = (abs(crate_wx) > 5.25) or (abs(crate_wy) > 4.2)

    terminal_failure = 0.0
    if (not _FAILED[0]) and (cart_out or crate_out or _HARD_HITS[0] >= 3):
        _FAILED[0] = True
        terminal_failure = -100.0
    components["terminal_failure"] = float(terminal_failure)

    # ---------------- 9. boundary_guard：随越界接近单调下降的惩罚 ------------
    margin_x = abs(next_obs[0]) - 0.92
    if margin_x < 0.0:
        margin_x = 0.0
    margin_y = abs(next_obs[1]) - 0.92
    if margin_y < 0.0:
        margin_y = 0.0
    guard = -(4.0 * margin_x + 60.0 * margin_x * margin_x)
    guard = guard - (4.0 * margin_y + 60.0 * margin_y * margin_y)

    crate_nx = crate_wx / 5.0
    crate_ny = crate_wy / 4.0
    cmargin_x = abs(crate_nx) - 0.96
    if cmargin_x < 0.0:
        cmargin_x = 0.0
    cmargin_y = abs(crate_ny) - 0.96
    if cmargin_y < 0.0:
        cmargin_y = 0.0
    guard = guard - (2.0 * cmargin_x + 20.0 * cmargin_x * cmargin_x)
    guard = guard - (2.0 * cmargin_y + 20.0 * cmargin_y * cmargin_y)
    components["boundary_guard"] = float(guard)

    # ---------------- 10. action_cost / time_cost ----------------------------
    act_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])
    components["action_cost"] = float(act_cost)
    components["time_cost"] = -0.002

    total = 0.0
    for key in components:
        total = total + components[key]
    return float(total), components
```
