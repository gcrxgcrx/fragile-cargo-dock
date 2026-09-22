# Response Record

```python
# ---------------------------------------------------------------------------
# 环境：俯视仓储推箱（cart 推 crate 入 dock）
# 观测：19 维，含义见环境事实。本函数只使用 obs / next_obs / action。
# 模块级状态用于：回合边界检测（obs[18] 单调递增，重置时回落）+ 一次性事件闩锁。
# ---------------------------------------------------------------------------
_PREV_T = [-1.0]        # 上一步的 time_fraction
_STREAK = [0]           # “泊位内 + 对齐 + 慢”的连续步数
_PAID = [False]         # terminal_success 是否已发放（整局一次）
_ENTERED = [False]      # 货箱是否曾经完全进入容差（dock_enter 整局一次）
_HARD_HITS = [0]        # 代理硬冲击计数（上升沿计一次）
_FAILED = [False]       # terminal_failure 是否已发放（整局一次）
_PREV_HARD = [False]    # 上一步是否处于“硬冲击”状态（用于上升沿检测）

# ---------------------------------------------------------------------------
# 自检记录（量级，单位：单步奖励）
#   正常推箱：cart 推 crate 以 ~0.8 m/s 朝泊位走，稳定接触（closing ≈ 0.05）
#     progress +0.80, approach ≈ 0.00, roughness -0.025, action_cost -0.0002,
#     time_cost -0.002  =>  R_push ≈ +0.77
#   什么都不做（货箱静止在初始位置）
#     R_idle ≈ -0.002
#   停稳在泊位内（容差内 + 对齐 + |v|<0.05）
#     settle_step +1.00, progress ≈ 0, time_cost -0.002  =>  R_settled ≈ +1.00
#   自检① R_push(0.77) > R_idle(-0.002)                                OK
#   自检② 悬停在泊位外 0.3 m、400 步 ≈ -0.8 ；真正入坞停稳(10 步+事件) ≈ +30  OK
#   自检③ ⑤接触+closing=1.0 => roughness -0.50 + hard_hit -0.50 = -1.00，
#          ⑥接触+closing=0.05 => roughness -0.025；差 ≈ 0.975，与推进项同量级  OK
#   自检④/⑤ 停稳状态连续调用：每步 settle_step=+1.0 恒发（不因 _PAID 关闭），
#          第 10 步额外 +300（环境单步裁剪为 20），随后环境立即终止。           OK
#   自检⑤ 边界：中心 (0,0) 附加 0；|cart_x|=1.05 附加 -1.5（明显更低）          OK
#   推进项两条都是有符号、对称的增量：靠近给正、远离给负。
# ---------------------------------------------------------------------------
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):

    # ---------------- 回合边界检测（必须，防止状态跨 episode 污染） ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 0.0025:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
        _FAILED[0] = False
        _PREV_HARD[0] = False
    _PREV_T[0] = t

    components = {}

    # ---------------- 几何还原（全部来自观测，按环境事实的尺度） --------------
    # 车->货箱：体坐标系相对位置（3.0 m 尺度），自身即真实距离
    rel_x = obs[6] * 3.0
    rel_y = obs[7] * 3.0
    d_cart_crate = (rel_x * rel_x + rel_y * rel_y) ** 0.5

    nrel_x = next_obs[6] * 3.0
    nrel_y = next_obs[7] * 3.0
    nd_cart_crate = (nrel_x * nrel_x + nrel_y * nrel_y) ** 0.5

    # 货箱->泊位：obs[12] 以半宽 5.0 m 归一，obs[13] 以半高 4.0 m 归一
    dx_m = obs[12] * 5.0
    dy_m = obs[13] * 4.0
    d_dock = (dx_m * dx_m + dy_m * dy_m) ** 0.5

    ndx_m = next_obs[12] * 5.0
    ndy_m = next_obs[13] * 4.0
    nd_dock = (ndx_m * ndx_m + ndy_m * ndy_m) ** 0.5

    # ---------------- 分项 1/2：两项有符号、对称的势函数增量 ----------------
    approach_cargo = d_cart_crate - nd_cart_crate   # >0: 本帧靠近货箱
    progress = d_dock - nd_dock                     # >0: 本帧货箱靠近泊位

    # ---------------- 分项 3：首次完全进入容差（整局一次） ----------------
    in_tol = 0.0
    if abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030:
        in_tol = 1.0
    dock_enter = 0.0
    if in_tol > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 5.0

    # ---------------- 分项 4：轻柔度代理（观测无冲量，用接近速度代理） ------
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0

    # 只在“接触中且正在接近”时为负；匀速推箱(closing≈0)几乎不罚。
    # k=0.5：closing=1.0 m/s 时 -0.50，与推进项(≈0.8/步)同量级；
    #        而正常推箱的 closing≈0.05 时仅 -0.025，不会压制推进。
    roughness = -0.5 * contact * closing

    # ---------------- 分项 7：硬冲击固定惩罚（代理 + 上升沿计数） ------------
    hard_now = False
    if contact > 0.5 and closing > 1.0:
        hard_now = True
    hard_hit = 0.0
    if hard_now and not _PREV_HARD[0]:
        _HARD_HITS[0] = _HARD_HITS[0] + 1
        hard_hit = -0.5
    _PREV_HARD[0] = hard_now

    # ---------------- 分项 5/6：动作代价与时间代价 ----------------
    action_cost = -0.0005 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.002

    # ---------------- 越界守卫（从 obs 自行判断边界，随接近单调下降） --------
    cart_nx = next_obs[0]
    cart_ny = next_obs[1]
    c = next_obs[2]
    s = next_obs[3]
    crate_xm = next_obs[0] * 5.0 + nrel_x * c - nrel_y * s
    crate_ym = next_obs[1] * 4.0 + nrel_x * s + nrel_y * c
    crate_nx = crate_xm / 5.0
    crate_ny = crate_ym / 4.0

    boundary_guard = 0.0
    over = abs(cart_nx) - 0.90          # 小车：>0.95 已明显生效
    if over > 0.0:
        boundary_guard = boundary_guard - 10.0 * over
    over = abs(cart_ny) - 0.90
    if over > 0.0:
        boundary_guard = boundary_guard - 10.0 * over
    over = abs(crate_nx) - 0.95         # 货箱：同样守卫，阈值略松
    if over > 0.0:
        boundary_guard = boundary_guard - 8.0 * over
    over = abs(crate_ny) - 0.95
    if over > 0.0:
        boundary_guard = boundary_guard - 8.0 * over

    # ---------------- 完成谓词（严格取自环境事实的容差与阈值） --------------
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    align = abs(next_obs[10])           # dock 轴与世界轴对齐，允许 90° 倍数
    if abs(next_obs[11]) > align:
        align = abs(next_obs[11])
    settled = False
    if in_tol > 0.5 and align >= 0.866 and crate_speed < 0.05:
        settled = True

    if settled:
        _STREAK[0] = _STREAK[0] + 1
    else:
        _STREAK[0] = 0

    # 分项 8：连续 10 步成立 -> 一次性事件（整局只发一次）
    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        terminal_success = 300.0

    # 停稳期每步收益：谓词成立就每步发放，绝不被 _PAID / 连续计数关闭。
    settle_step = 1.0 if settled else 0.0

    # ---------------- 分项 9：越界 或 累计硬冲击>=3 -> 一次性失败惩罚 --------
    out_of_bounds = False
    if abs(cart_nx) > 1.05 or abs(cart_ny) > 1.05:
        out_of_bounds = True
    if abs(crate_nx) > 1.05 or abs(crate_ny) > 1.05:
        out_of_bounds = True

    terminal_failure = 0.0
    if not _FAILED[0] and (out_of_bounds or _HARD_HITS[0] >= 3):
        _FAILED[0] = True
        terminal_failure = -100.0

    # ---------------- 汇总 ----------------
    components['approach_cargo'] = approach_cargo
    components['progress'] = progress
    components['dock_enter'] = dock_enter
    components['roughness'] = roughness
    components['action_cost'] = action_cost
    components['time_cost'] = time_cost
    components['hard_hit'] = hard_hit
    components['boundary_guard'] = boundary_guard
    components['settle_step'] = settle_step
    components['terminal_success'] = terminal_success
    components['terminal_failure'] = terminal_failure

    total = (approach_cargo
             + progress
             + dock_enter
             + roughness
             + action_cost
             + time_cost
             + hard_hit
             + boundary_guard
             + settle_step
             + terminal_success
             + terminal_failure)

    return float(total), components
```
