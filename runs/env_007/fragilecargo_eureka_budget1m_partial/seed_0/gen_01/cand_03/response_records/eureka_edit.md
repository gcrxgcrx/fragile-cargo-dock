# Response Record

## 分析

1. `docking_settle_success` 均值 71.2（最大 678），总奖励 72.85 ≈ 它，说明**策略几乎只在优化这个停稳代理**；而 19/20 episode 是 truncated、任务分仅 18.67，证明代理太松（`near` 阈值 1.5 m、`speed` 阈值 0.6 m/s 远离真实 0.05 m/s），agent 在 dock 附近低速悬停刷分而不真正入位。
2. 主进展信号被淹没：`crate_progress_toward_dock` 均值仅 1.84，净推进几乎停滞。
3. `fragile_handling_guard` 均值 -0.0015，几乎不触发，可保留但需放低阈值备用。
4. 修改：把停稳代理**大幅收紧并改回裸乘积**（近 0.5 m × 慢 0.12 m/s × 对齐 <30°），权重降到与进展同级；提高主进展权重；新增无接触期的接近货箱引导；保留轻量安全 hinge。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    components = {}

    # ---------- 观测量反解（严格使用已声明索引） ----------
    dx = obs[12] * 5.0
    dy = obs[13] * 4.0
    dist = (dx * dx + dy * dy) ** 0.5

    ndx = next_obs[12] * 5.0
    ndy = next_obs[13] * 4.0
    next_dist = (ndx * ndx + ndy * ndy) ** 0.5

    crate_vx = obs[8] * 3.0
    crate_vy = obs[9] * 3.0
    n_vx = next_obs[8] * 3.0
    n_vy = next_obs[9] * 3.0
    next_crate_speed = (n_vx * n_vx + n_vy * n_vy) ** 0.5

    contact = obs[14]

    # ---------- 1. 货箱向 dock 的净进展（主稠密学习信号，delta 抑制悬停） ----------
    delta = dist - next_dist
    if delta > 1.5:
        delta = 1.5
    elif delta < -1.5:
        delta = -1.5
    adelta = delta if delta > 0.0 else -delta
    components["crate_progress_toward_dock"] = 4.0 * (delta / (1.0 + adelta))

    # ---------- 2. 无接触阶段接近货箱的导航引导（delta，防绕圈） ----------
    rx = obs[6] * 3.0
    ry = obs[7] * 3.0
    cart_crate = (rx * rx + ry * ry) ** 0.5
    nrx = next_obs[6] * 3.0
    nry = next_obs[7] * 3.0
    n_cart_crate = (nrx * nrx + nry * nry) ** 0.5
    if contact < 0.5:
        d2 = cart_crate - n_cart_crate
        if d2 > 1.5:
            d2 = 1.5
        elif d2 < -1.5:
            d2 = -1.5
        ad2 = d2 if d2 > 0.0 else -d2
        components["approach_crate"] = 1.0 * (d2 / (1.0 + ad2))
    else:
        components["approach_crate"] = 0.0

    # ---------- 3. 停稳完成近似：严格贴近真实成功条件（近 + 慢 + 对齐） ----------
    near = 1.0 - next_dist / 0.5
    if near < 0.0:
        near = 0.0
    elif near > 1.0:
        near = 1.0

    speed_factor = 1.0 - next_crate_speed / 0.12
    if speed_factor < 0.0:
        speed_factor = 0.0
    elif speed_factor > 1.0:
        speed_factor = 1.0

    cc = next_obs[10]
    ss = next_obs[11]
    ac = cc if cc >= 0.0 else -cc
    asn = ss if ss >= 0.0 else -ss
    align = ac if ac > asn else asn
    align_factor = (align - 0.70710678) / 0.15891862
    if align_factor < 0.0:
        align_factor = 0.0
    elif align_factor > 1.0:
        align_factor = 1.0

    components["docking_settle_success"] = 4.0 * near * speed_factor * align_factor

    # ---------- 4. 脆弱货箱保护（接触瞬间高速接近 = 硬碰撞代理，hinge 惩罚） ----------
    approach = 0.0
    if contact > 0.5:
        ch = obs[2]
        sh = obs[3]
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
    components["fragile_handling_guard"] = -1.0 * (overspeed ** 2)

    # ---------- 5. 越界防护（小车与货箱均不得离开地板，hinge 仅近边界生效） ----------
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
