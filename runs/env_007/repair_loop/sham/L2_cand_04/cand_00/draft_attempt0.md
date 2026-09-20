分析：当前奖励的组件统计严重错乱（多组件的 mean/abs_mean/min/max 与 activation 互相矛盾，说明日志本身异常），但可确定三点：①`progress` 用 delta 距离且只在接近时给分，激活率极低，主信号太稀疏，货箱几乎不前进；②`gentleness` 与 `speed_gate_penalty` 用"速度/接近速度"作惩罚，会压制推动动作，且量级压过 progress；③`success_event`/`entry_bonus` 的 one-shot 在稀疏主信号下几乎不触发，episode 全部 truncation，任务分数为负。改法：把主信号改为稠密的"货箱到泊位距离势能"（只奖励靠近、不惩罚远离，保证推动动作净正），用联合连续 proxy 引导"进入+对齐+静止"，把速度/接触惩罚改成仅在泊位内触发的门控惩罚（不影响接近过程），并去掉会惩罚推动的 gentleness。

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    _PREV_T = [-1.0]
    _STREAK = [0]
    _PAID = [False]
    _ENTERED = [False]

    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    ch = float(next_obs[10])
    sh = float(next_obs[11])
    heading_err = (sh * sh) ** 0.5

    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    inside = 1.0 if (dx <= 0.024 and dx >= -0.024 and dy <= 0.030 and dy >= -0.030) else 0.0
    aligned = 1.0 if heading_err < 0.5 else 0.0
    slow = 1.0 if crate_speed < 0.05 else 0.0
    complete_state = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    if complete_state > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    components = {}

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0

    entry_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        entry_bonus = 20.0

    if complete_state > 0.5:
        components["success_event"] = success_event
        components["entry_bonus"] = entry_bonus
        components["progress"] = 0.0
        components["dock_quality"] = 0.0
        components["speed_gate_penalty"] = 0.0
        components["bounds_penalty"] = 0.0
        components["action_penalty"] = 0.0
        total = success_event + entry_bonus
        return (float(total), components)

    # ---------- dense potential-style progress (only rewards approach) ----------
    # Phi = 1 / (1 + k * dist): bounded, monotone in distance, always positive gradient
    # toward the dock. Rewards pushing the crate closer; never penalizes motion.
    k_dist = 6.0
    phi_now = 1.0 / (1.0 + k_dist * dist)
    # baseline potential at worst-case distance (dist ~ 2.0) so it is not a constant offset
    phi_ref = 1.0 / (1.0 + k_dist * 2.0)
    progress = 40.0 * (phi_now - phi_ref)

    # ---------- joint soft proxy: inside & aligned & slow (continuous) ----------
    # inside factor: 1 at center, decays to 0 at ~2x the dock half-extents
    fx = 1.0 - abs(dx) / 0.10
    if fx < 0.0:
        fx = 0.0
    fy = 1.0 - abs(dy) / 0.12
    if fy < 0.0:
        fy = 0.0
    f_inside = (fx * fy) ** 0.5
    # alignment: 1 when aligned, decays by heading error
    f_align = 1.0 - heading_err / 0.9
    if f_align < 0.0:
        f_align = 0.0
    # slowness: 1 when still, decays with speed
    f_slow = 1.0 - crate_speed / 0.6
    if f_slow < 0.0:
        f_slow = 0.0
    dock_quality = 25.0 * (f_inside * f_align * f_slow) ** (1.0 / 3.0)

    # ---------- speed gate penalty: only when already inside the dock region ----------
    # (does not penalize approach; only suppresses residual speed once crate is in the dock)
    speed_gate_penalty = 0.0
    if inside > 0.5 and crate_speed > 0.05:
        over = crate_speed - 0.05
        speed_gate_penalty = -8.0 * over * over

    # ---------- out-of-bounds guard (cart) ----------
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    ax = cx if cx >= 0.0 else -cx
    ay = cy if cy >= 0.0 else -cy
    bounds_penalty = 0.0
    if ax > 0.90:
        bounds_penalty -= 40.0 * (ax - 0.90)
    if ay > 0.90:
        bounds_penalty -= 40.0 * (ay - 0.90)

    # ---------- light action smoothness ----------
    a0 = float(action[0])
    a1 = float(action[1])
    action_penalty = -0.02 * (a0 * a0 + a1 * a1)

    components["success_event"] = success_event
    components["entry_bonus"] = entry_bonus
    components["progress"] = progress
    components["dock_quality"] = dock_quality
    components["speed_gate_penalty"] = speed_gate_penalty
    components["bounds_penalty"] = bounds_penalty
    components["action_penalty"] = action_penalty

    total = (success_event + entry_bonus + progress + dock_quality
             + speed_gate_penalty + bounds_penalty + action_penalty)
    return (float(total), components)