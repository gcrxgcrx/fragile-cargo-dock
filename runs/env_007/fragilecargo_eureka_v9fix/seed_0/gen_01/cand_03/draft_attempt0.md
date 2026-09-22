问题：总回报在训练全程停在 17~18，唯一显著的学习信号是稀疏的一次性成功奖励（terminal_success 从 0 涨到 130~230），而占据 99% 步数的推进项 progress 每回合仅约 2.7，梯度被巨量稀疏奖励淹没；评测 19/20 回合超时截断，说明最终阶段学不会。成功还要求货箱朝向对齐与低速，但当前奖励里对齐完全没有密集项、低速只在"完全入坞后"以 0/1 二值出现，缺少可爬的坡度。另外 boundary_guard/terminal_failure 量级过大（单回合曾达 -414），带来高方差。改法：保留基于距离差分的 progress（望远镜求和，不会诱导悬停），把二值 settle_bonus 换成"包含度 × 对齐 × 低速"的连续联合代理，新增按坞距门控的连续对齐项，下调一次性成功/失败/越界量级，并去掉无效的常数时间成本量级。

```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_HARD_HITS = [0]
_FAILED_PAID = [False]

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 自检：①静止不动（货箱停在初始位置、远离 dock）单步 ≈ -0.001（仅极小时间成本，settle/align 门控为 0）
    #       ②正把货箱推向 dock（货箱因此有速度，仍在 dock 外）单步 ≈ +1.5*Δd - 0.002 > ①，严格占优。
    #       进入 dock 后 settle 项才随"包含度/对齐/低速"连续上升，上限 +1.0/步，不会对推进产生净负贡献。

    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _HARD_HITS[0] = 0
        _FAILED_PAID[0] = False
    _PREV_T[0] = t

    # --- 小车→货箱接近（帮助早期找到货箱） ---
    prev_cart_crate = ((obs[6] * 3.0) ** 2 + (obs[7] * 3.0) ** 2) ** 0.5
    next_cart_crate = ((next_obs[6] * 3.0) ** 2 + (next_obs[7] * 3.0) ** 2) ** 0.5
    approach_cargo = 0.5 * (prev_cart_crate - next_cart_crate)

    # --- 货箱→dock 距离差分（望远镜求和，稳态悬停收益为 0） ---
    prev_dock = ((obs[12] * 5.0) ** 2 + (obs[13] * 4.0) ** 2) ** 0.5
    next_dock = ((next_obs[12] * 5.0) ** 2 + (next_obs[13] * 4.0) ** 2) ** 0.5
    progress = 1.5 * (prev_dock - next_dock)

    # --- 货箱航向与仓库轴的对齐度（方形货箱允许 0/90 度，取 max(|cos|,|sin|)） ---
    align01 = abs(next_obs[10])
    if abs(next_obs[11]) > align01:
        align01 = abs(next_obs[11])
    if align01 > 1.0:
        align01 = 1.0

    # --- 首次完全进入 dock 容差的一次性奖励 ---
    in_dock = (abs(next_obs[12]) <= 0.024 and abs(next_obs[13]) <= 0.030)
    dock_enter = 0.0
    if in_dock and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = 3.0

    # --- 靠近 dock 时才激活的连续对齐引导（远离时为 0，不影响推动阶段） ---
    near_gate = 1.0 - next_dock / 1.2
    if near_gate < 0.0:
        near_gate = 0.0
    align_near = 0.08 * near_gate * align01

    # --- 低速：货箱世界速度大小 ---
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    speed = (crate_vx ** 2 + crate_vy ** 2) ** 0.5

    # --- 联合完成代理：连续几何量，替代原来的 0/1 settle_bonus ---
    e_pos = abs(next_obs[12]) / 0.024
    e_pos_y = abs(next_obs[13]) / 0.030
    if e_pos_y > e_pos:
        e_pos = e_pos_y
    f_pos = (2.0 - e_pos) / 1.0
    if f_pos < 0.0:
        f_pos = 0.0
    if f_pos > 1.0:
        f_pos = 1.0

    f_align = align01

    f_slow = (0.5 - speed) / 0.45
    if f_slow < 0.0:
        f_slow = 0.0
    if f_slow > 1.0:
        f_slow = 1.0

    settle = 1.0 * f_pos * (0.5 * f_align + 0.5 * f_slow)

    # --- 接触轻柔度代理：只有相对逼近速度（真正会造成冲击的分量）才被惩罚 ---
    crate_along = crate_vx * next_obs[2] + crate_vy * next_obs[3]
    closing = next_obs[4] * 3.0 - crate_along
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    roughness = -0.3 * contact * closing

    hard_hit = 0.0
    if contact > 0.5 and closing > 1.0:
        hard_hit = -0.4
        _HARD_HITS[0] += 1

    # --- 动作与时间成本（量级远小于主信号） ---
    action_cost = -0.0006 * (action[0] ** 2 + action[1] ** 2)
    time_cost = -0.001

    # --- 完成谓词与连续保持计数 ---
    done_now = in_dock and (align01 >= 0.8660254) and (speed < 0.05)
    if done_now:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_bonus = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_bonus = 200.0

    # --- 边界守卫（软约束，量级收敛） ---
    pen_x = max(0.0, abs(next_obs[0]) - 0.9)
    pen_y = max(0.0, abs(next_obs[1]) - 0.9)
    boundary_guard = -20.0 * (pen_x + pen_y)

    # --- 失败一次项：小车/货箱越界，或累计 3 次硬冲击 ---
    terminal_failure = 0.0
    if not _FAILED_PAID[0]:
        cart_out = (abs(next_obs[0]) > 1.02 or abs(next_obs[1]) > 1.02)

        cart_x = next_obs[0] * 5.0
        cart_y = next_obs[1] * 4.0
        cos_h = next_obs[2]
        sin_h = next_obs[3]
        rel_x = next_obs[6] * 3.0
        rel_y = next