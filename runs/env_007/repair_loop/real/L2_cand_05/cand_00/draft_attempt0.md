**分析**：任务分数 -1.33、20/20 全部 truncation，说明策略从未接近成功；`progress` 激活率仅 0.1%~12% 且快速衰减到 0，货箱根本没被推动（`delta>0` 太稀疏，且只在"更接近"时给分，一旦原地打转就零梯度）；`dock_quality`、`speed_pen`、`enter_event`、`success_event` 全程为 0，说明这些信号要么门控太严（`dist<0.15`）要么二值化过稀。`bound_pen` 与 `gentleness` 是仅有的持续负项，尤其 gentleness 在早期占了主导，策略被"别碰箱"引导成不动。改进方向：① 用**连续势能**（距离本身的有界衰减）替代稀疏 delta，让每步都有指向泊位的梯度；② 把"进入泊位 + 朝向 + 低速"做成**连续联合代理**（几何平均），全程可微；③ 把速度/朝向做成**接近泊位时才启用的门控**，且不惩罚"推动所必需"的速度；④ 轻柔惩罚改为只在**高速接触**时触发，避免压制正常推动；⑤ 边界惩罚用 hinge 且降权。

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode 边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 货箱到泊位偏移（归一化） ----------
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    odx = float(obs[12])
    ody = float(obs[13])
    old_dist = (odx * odx + ody * ody) ** 0.5

    # ---------- 货箱速度 ----------
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---------- 朝向误差代理（货箱朝向 vs +x 轴） ----------
    cc = float(next_obs[10])
    cs = float(next_obs[11])
    if cc > 0.0:
        ang = cs
    else:
        ang = 1.0 if cs >= 0.0 else -1.0
    ang_err = abs(ang)  # 0=对齐, 1=垂直

    # ---------- 完成条件（容差取自环境事实） ----------
    in_tol = (abs(dx) <= 0.024) and (abs(dy) <= 0.030)
    aligned = ang_err < 0.5
    slow = crate_speed < 0.05
    complete_state = in_tol and aligned and slow

    # ---------- 边界守卫（hinge，只在接近边界时生效） ----------
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    ax = abs(cx)
    ay = abs(cy)
    bound_pen = 0.0
    if ax > 0.98:
        bound_pen -= 3.0 * (ax - 0.98)
    if ay > 0.98:
        bound_pen -= 3.0 * (ay - 0.98)

    # ---------- 轻柔度：仅在高速接触时惩罚（避免压制正常推动） ----------
    crate_along_heading = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    # 只在接近速度超过 0.6 m/s 时惩罚（hard hinge），正常推动不触发
    hard_close = closing - 0.6
    if hard_close < 0.0:
        hard_close = 0.0
    gentleness = -5.0 * contact * hard_close

    # ---------- 连续势能：货箱到泊位的接近信号（每步都有梯度） ----------
    # 用有界衰减：dist=0 -> 1, dist 越大越接近 0
    proximity = 1.0 / (1.0 + 8.0 * dist)
    # 增量奖励：靠近时正、远离时轻微负，保证"推向泊位"净正
    delta = old_dist - dist
    progress = 30.0 * delta + 2.0 * proximity

    # ---------- 连续联合代理：进入 + 朝向 + 低速（几何平均，全程可微） ----------
    # 位置因子：越接近泊位中心越接近 1
    f_pos = 1.0 / (1.0 + 60.0 * dist)
    # 朝向因子：ang_err=0 -> 1, ang_err=0.5 -> 0
    f_align = 1.0 - ang_err / 0.5
    if f_align < 0.0:
        f_align = 0.0
    # 速度因子：速度=0 -> 1, 速度=0.05 -> 0
    f_slow = 1.0 - crate_speed / 0.05
    if f_slow < 0.0:
        f_slow = 0.0
    # 几何平均（避免乘积塌缩）
    dock_quality = 6.0 * (f_pos * f_align * f_slow) ** (1.0 / 3.0)

    # ---------- 接近泊位时的速度抑制（门控：只惩罚"滑过泊位"风险） ----------
    speed_pen = 0.0
    if dist < 0.20:
        # 只在货箱已接近泊位且速度仍高时抑制，且不惩罚推动所必需的合理速度
        over = crate_speed - 0.4
        if over > 0.0:
            speed_pen = -2.0 * over * (0.20 - dist) / 0.20

    # ---------- 完成事件与持续检测 ----------
    if complete_state:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 200.0

    enter_event = 0.0
    if in_tol and aligned and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 10.0

    total = progress + dock_quality + speed_pen + gentleness + bound_pen + success_event + enter_event

    components = {
        "progress": progress,
        "dock_quality": dock_quality,
        "speed_pen": speed_pen,
        "gentleness": gentleness,
        "bound_pen": bound_pen,
        "enter_event": enter_event,
        "success_event": success_event,
    }
    return float(total), components