```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- episode 边界检测（obs[18] 单调递增，重置时回落）----
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---- 货箱到泊位的有符号偏移（归一化）----
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    # 位置容差（环境事实：|dx|<=0.024 且 |dy|<=0.030 才算完全进入）
    in_dock_pos = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0

    # ---- 货箱速度（世界系，恢复到 m/s）----
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---- 货箱朝向误差（rad）----
    cc = float(next_obs[10])
    cs = float(next_obs[11])
    ang_err = abs(_atan2(cc, cs))

    # ---- 完成条件（显式从 obs 推断，容差取自环境事实）----
    cond_pos = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0
    cond_ang = 1.0 if ang_err < 0.5236 else 0.0      # 30°
    cond_vel = 1.0 if crate_speed < 0.05 else 0.0
    if cond_pos > 0.5 and cond_ang > 0.5 and cond_vel > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # ---- 组件 1：接近进度（增量形式，只在更接近时给分）----
    prev_dist = (float(obs[12]) ** 2 + float(obs[13]) ** 2) ** 0.5
    progress = prev_dist - dist
    if progress < 0.0:
        progress = 0.0
    r_progress = 20.0 * progress

    # ---- 组件 2：轻柔接触（唯一教会"接近泊位时减速"的信号）----
    crate_along_heading = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    r_gentle = -0.05 * contact * closing

    # ---- 组件 3：泊位内对齐与静止 shaping（门控在"已进入泊位"上，非全局）----
    if in_dock_pos > 0.5:
        align = 1.0 - ang_err / 0.5236
        if align < 0.0:
            align = 0.0
        slow = 1.0 - crate_speed / 0.5
        if slow < 0.0:
            slow = 0.0
        r_dock_quality = 0.5 * align + 0.5 * slow
    else:
        r_dock_quality = 0.0

    # ---- 组件 4：越界 hinge 惩罚 ----
    cart_x = float(next_obs[0])
    cart_y = float(next_obs[1])
    oob = 0.0
    if abs(cart_x) > 0.95:
        oob += abs(cart_x) - 0.95
    if abs(cart_y) > 0.95:
        oob += abs(cart_y) - 0.95
    r_oob = -2.0 * oob

    # ---- 组件 5：一次性进入泊位奖励（整局一次，帮助跨过"进得去停不住"）----
    enter_event = 0.0
    if in_dock_pos > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_event = 5.0

    # ---- 组件 6：一次性完成事件（连续 10 步满足，整局只发一次）----
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    total = (r_progress + r_gentle + r_dock_quality + r_oob
             + enter_event + success_event)

    components = {
        "crate_to_dock_progress": r_progress,
        "soft_contact_penalty": r_gentle,
        "docking_quality": r_dock_quality,
        "out_of_bounds_penalty": r_oob,
        "enter_dock_event": enter_event,
        "success_event": success_event,
    }
    return float(total), components


def _atan2(y, x):
    # 简易 atan2，避免 import math
    if x > 0.0:
        return _atan(y / x)
    if x < 0.0:
        if y >= 0.0:
            return _atan(y / x) + 3.141592653589793
        return _atan(y / x) - 3.141592653589793
    if y > 0.0:
        return 1.5707963267948966
    if y < 0.0:
        return -1.5707963267948966
    return 0.0


def _atan(z):
    # 简易 atan，|z| 可大于 1，用恒等式收敛
    if z > 1.0:
        return 1.5707963267948966 - _atan_core(1.0 / z)
    if z < -1.0:
        return -1.5707963267948966 + _atan_core(-1.0 / z)
    return _atan_core(z)


def _atan_core(z):
    # |z| <= 1 的泰勒级数
    zz = z * z
    term = z
    result = z
    n = 1
    while n < 20:
        term = -term * zz
        result += term / (2 * n + 1)
        n += 1
    return result


_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
```