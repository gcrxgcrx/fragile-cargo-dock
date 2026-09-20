```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [-1.0]
_LAST_ACTION = [0.0, 0.0]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode 边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
        _LAST_ACTION[0] = 0.0
        _LAST_ACTION[1] = 0.0
    _PREV_T[0] = t

    # ---------- 几何量（由 obs 恢复） ----------
    # 货箱到泊位偏移（归一化）
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    # 泊位容差：|obs[12]| <= 0.024, |obs[13]| <= 0.030
    dist = (dx * dx + dy * dy) ** 0.5

    # 货箱速度（世界系，m/s）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 朝向误差（货箱朝向 vs 泊位朝向，泊位朝向取 +x 方向即 cos=1,sin=0）
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    # 朝向误差角（弧度），via atan2
    angle_err = abs(_atan2(sh, ch))
    # 30° 容差
    angle_ok = 1.0 if angle_err < 0.5235987755982988 else 0.0

    # 接触与接近速度代理
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0

    # ---------- 完成条件（必须严格从 obs 推断） ----------
    inside = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0
    slow = 1.0 if crate_speed < 0.05 else 0.0
    aligned = angle_ok
    complete_now = 1.0 if (inside > 0.5 and slow > 0.5 and aligned > 0.5) else 0.0

    if complete_now > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    # 首次进入泊位（一次性）
    entered_event = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        entered_event = 40.0

    # ---------- 组件 1：货箱向泊位的推进（增量形式，避免悬停收割） ----------
    if _PREV_DIST[0] < 0.0:
        _PREV_DIST[0] = dist
    delta_dist = _PREV_DIST[0] - dist
    _PREV_DIST[0] = dist
    # 增量奖励，限制幅度防止单步爆炸
    if delta_dist > 0.05:
        delta_dist = 0.05
    if delta_dist < -0.05:
        delta_dist = -0.05
    progress = 20.0 * delta_dist

    # ---------- 组件 2：接近泊位时的速度抑制（门控，仅近距离激活） ----------
    # 距离门：越近越强，1.0 在 0.15 归一化距离外为 0
    if dist < 0.15:
        near_gate = (0.15 - dist) / 0.15
    else:
        near_gate = 0.0
    if near_gate < 0.0:
        near_gate = 0.0
    # 只在接近泊位时对货箱速度做抑制（二次）
    speed_near_pen = -0.30 * near_gate * (crate_speed ** 2)

    # ---------- 组件 3：轻柔接触（唯一能教会减速的信号） ----------
    gentleness = -0.05 * contact * closing

    # ---------- 组件 4：朝向对齐 shaping（仅在靠近泊位时给正分，增量式） ----------
    # 用连续 bounded 形式，只在 dist 小时激活
    if dist < 0.20:
        align_gate = (0.20 - dist) / 0.20
    else:
        align_gate = 0.0
    if align_gate < 0.0:
        align_gate = 0.0
    # 朝向误差连续因子：1 表示完美对齐，0 表示 >=90°
    if angle_err < 1.5707963267948966:
        align_factor = 1.0 - (angle_err / 1.5707963267948966)
    else:
        align_factor = 0.0
    align_shaping = 0.5 * align_gate * align_factor

    # ---------- 组件 5：越界 hinge 惩罚 ----------
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    if abs(cart_x) > 0.95:
        oob -= 2.0 * (abs(cart_x) - 0.95)
    if abs(cart_y) > 0.95:
        oob -= 2.0 * (abs(cart_y) - 0.95)
    # 货箱越界：用归一化偏移推断（泊位中心离仓库边界有一定距离，这里只做软约束）
    if abs(dx) > 0.9:
        oob -= 2.0 * (abs(dx) - 0.9)
    if abs(dy) > 0.9:
        oob -= 2.0 * (abs(dy) - 0.9)

    # ---------- 组件 6：动作平滑（轻量，避免过度抑制） ----------
    da0 = float(action[0]) - _LAST_ACTION[0]
    da1 = float(action[1]) - _LAST_ACTION[1]
    _LAST_ACTION[0] = float(action[0])
    _LAST_ACTION[1] = float(action[1])
    smooth = -0.02 * (da0 * da0 + da1 * da1)

    # ---------- 汇总 ----------
    components = {
        "progress": progress,
        "speed_near_dock": speed_near_pen,
        "gentleness": gentleness,
        "align_shaping": align_shaping,
        "out_of_bounds": oob,
        "action_smoothness": smooth,
        "entered_event": entered_event,
        "success_event": success_event,
    }
    total = (progress + speed_near_pen + gentleness + align_shaping
             + oob + smooth + entered_event + success_event)
    return float(total), components


def _atan2(y, x):
    # 简易 atan2，避免 import math
    if x > 0.0:
        return _atan(y / x)
    elif x < 0.0:
        if y >= 0.0:
            return _atan(y / x) + 3.141592653589793
        else:
            return _atan(y / x) - 3.141592653589793
    else:
        if y > 0.0:
            return 1.5707963267948966
        elif y < 0.0:
            return -1.5707963267948966
        else:
            return 0.0


def _atan(z):
    # 简易 atan 近似（|z| 通过恒等式压缩到 <=1）
    if z > 1.0:
        return 1.5707963267948966 - _atan_core(1.0 / z)
    elif z < -1.0:
        return -1.5707963267948966 - _atan_core(1.0 / z)
    else:
        return _atan_core(z)


def _atan_core(z):
    # 多项式近似，|z| <= 1
    z2 = z * z
    return z * (0.9998660 + z2 * (-0.3302995 + z2 * (0.1801410 + z2 * (-0.0851330 + z2 * 0.0208351))))
```