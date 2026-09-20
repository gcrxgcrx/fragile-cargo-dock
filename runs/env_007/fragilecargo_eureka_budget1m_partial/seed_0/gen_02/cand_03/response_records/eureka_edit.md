# Response Record

## 分析

1. `docking_settle_success` 单组件 episode 和约 71（总奖励 73），远压过主进展信号，而它的阈值太松（1.5 m / 0.6 m·s⁻¹ / 45°），等于奖励“把货箱停在 dock 附近并保持慢速与近似轴向”，策略因此学会了悬停刷分——19/20 episode 被截断、只有 1 次终止，任务分数停滞。
2. `crate_progress_toward_dock` 的 mean≈abs mean≈1.9，说明货箱几乎只在原地微动，主学习信号被悬停行为淹没；需要提高其权重并保留 bounded delta 的形式。
3. `fragile_handling_guard` 几乎不触发（-0.0015，阈值 0.5 m·s⁻¹ 偏高），无法起到“轻接触”塑形作用，应下调阈值使其成为真实的柔和接触信号。
4. 新增两个轻量约束：远离 dock 且货箱近乎静止时的 hinge 停滞惩罚（打破悬停均衡），以及货箱高速滑行惩罚（降低冲进 dock 时超速/斜向的风险）。
5. 把停靠奖励收紧为真正的“入库”近似：距离门改为 1.0 m 的三次方衰减、速度满分要求 ≤0.05 m·s⁻¹、对齐满分约 20°（30° 时已衰减到 0.39），从根本上消除刷分。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    components = {}

    # ---- 反解可观测量（严格使用已声明索引） ----
    crate_dx = obs[12] * 5.0
    crate_dy = obs[13] * 4.0
    dist = (crate_dx * crate_dx + crate_dy * crate_dy) ** 0.5

    n_dx = next_obs[12] * 5.0
    n_dy = next_obs[13] * 4.0
    next_dist = (n_dx * n_dx + n_dy * n_dy) ** 0.5

    crate_vx = obs[8] * 3.0
    crate_vy = obs[9] * 3.0

    n_vx = next_obs[8] * 3.0
    n_vy = next_obs[9] * 3.0
    next_crate_speed = (n_vx * n_vx + n_vy * n_vy) ** 0.5

    # ---- 组件 1：货箱向 dock 的净进展（主稠密信号；delta 抑制“停在附近”刷分） ----
    delta = dist - next_dist
    if delta > 2.0:
        delta = 2.0
    elif delta < -2.0:
        delta = -2.0
    components["crate_progress_toward_dock"] = 3.0 * (delta / (1.0 + abs(delta)))

    # ---- 组件 2：真正“入库并停稳”的联合条件（收紧到成功判据附近） ----
    # 距离门：1.0 m 内三次方衰减，仅当货箱基本落入 dock 才给出显著值
    t_dist = 1.0 - next_dist / 1.0
    if t_dist < 0.0:
        t_dist = 0.0
    elif t_dist > 1.0:
        t_dist = 1.0
    near = t_dist * t_dist * t_dist

    # 速度门：<=0.05 m/s 满分，0.25 m/s 归零（与“近静止 <0.05”判据对齐）
    t_speed = (0.25 - next_crate_speed) / 0.20
    if t_speed < 0.0:
        t_speed = 0.0
    elif t_speed > 1.0:
        t_speed = 1.0

    # 朝向门：以最近轴向为基准，约 20° 内满分，35° 归零（成功判据为 30°）
    cc = next_obs[10]
    ss = next_obs[11]
    ac = cc if cc >= 0.0 else -cc
    asn = ss if ss >= 0.0 else -ss
    align = ac if ac > asn else asn
    t_align = (align - 0.81915204) / 0.12074796
    if t_align < 0.0:
        t_align = 0.0
    elif t_align > 1.0:
        t_align = 1.0

    components["docking_settle_success"] = 3.0 * (near * t_speed * t_align)

    # ---- 组件 3：脆弱货箱保护（接触瞬间高速接近 = 硬碰撞代理，hinge 惩罚） ----
    contact = obs[14]
    approach = 0.0
    if contact > 0.5:
        ch = obs[2]
        sh = obs[3]
        rx = obs[6] * 3.0
        ry = obs[7] * 3.0
        wx = rx * ch - ry * sh
        wy = rx * sh + ry * ch
        nrm = (wx * wx + wy * wy) ** 0.5
        if nrm > 1e-6:
            nx_dir = wx / nrm
            ny_dir = wy / nrm
            cart_vx = ch * (obs[4] * 3.0)
            cart_vy = sh * (obs[4] * 3.0)
            rel_vx = cart_vx - crate_vx
            rel_vy = cart_vy - crate_vy
            approach = rel_vx * nx_dir + rel_vy * ny_dir
        if approach < 0.0:
            approach = 0.0
    overspeed = approach - 0.4
    if overspeed < 0.0:
        overspeed = 0.0
    components["fragile_handling_guard"] = -1.2 * (overspeed * overspeed)

    # ---- 组件 4：货箱高速滑行约束（降低冲进 dock 时超速/斜向停不稳的风险） ----
    crate_fast = next_crate_speed - 1.8
    if crate_fast < 0.0:
        crate_fast = 0.0
    components["crate_overspeed_guard"] = -0.25 * (crate_fast * crate_fast)

    # ---- 组件 5：运输段停滞抑制（仅在远离 dock 且货箱近乎静止时生效） ----
    far = (next_dist - 0.8) / 1.2
    if far < 0.0:
        far = 0.0
    elif far > 1.0:
        far = 1.0
    still = 1.0 - next_crate_speed / 0.08
    if still < 0.0:
        still = 0.0
    elif still > 1.0:
        still = 1.0
    components["transit_stall_guard"] = -0.08 * (far * still)

    # ---- 组件 6：越界防护（小车与货箱都不得离开仓库地板，hinge 只在地板边缘附近生效） ----
    gx = next_obs[0]
    if gx < 0.0:
        gx = -gx
    gy = next_obs[1]
    if gy < 0.0:
        gy = -gy
    cart_over = gx - 0.9
    if cart_over < 0.0:
        cart_over = 0.0
    cart_over_y = gy - 0.9
    if cart_over_y < 0.0:
        cart_over_y = 0.0

    nch = next_obs[2]
    nsh = next_obs[3]
    rxb = next_obs[6] * 3.0
    ryb = next_obs[7] * 3.0
    crate_wx = next_obs[0] * 5.0 + (rxb * nch - ryb * nsh)
    crate_wy = next_obs[1] * 4.0 + (rxb * nsh + ryb * nch)
    cbx = crate_wx / 5.0
    if cbx < 0.0:
        cbx = -cbx
    cby = crate_wy / 4.0
    if cby < 0.0:
        cby = -cby
    crate_over = cbx - 0.9
    if crate_over < 0.0:
        crate_over = 0.0
    crate_over_y = cby - 0.9
    if crate_over_y < 0.0:
        crate_over_y = 0.0

    components["bounds_guard"] = -0.5 * (cart_over + cart_over_y + crate_over + crate_over_y)

    total_reward = 0.0
    for key in components:
        total_reward = total_reward + components[key]
    return (float(total_reward), components)
```
