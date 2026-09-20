## 分析

`docking_settle` 以 147/167（≈88%）压过其余所有组件，而 `crate_progress` 的 mean≈abs mean 说明货箱位移来回振荡、净进展很小，加上 18/20 截断、任务分仅 33.7 —— 典型的"代理-目标脱节"：策略在 dock 附近低速悬停收割稠密状态分，而不是真正完成停靠。因此要让**望远镜式（telescoping）的净进度**成为主信号（提高权重、无悬停农分），把 `docking_settle` 从稠密大额状态分改造为**收紧阈值 × 乘积形式 × 低权重**的完成度探针，使其只在接近"真停靠"（近中心+近静止+轴向对齐）时才显著。同时把硬接触相对速度阈值下调、惩罚略增，强化"轻拿轻放"约束；边界惩罚保留。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 主信号：货箱向 dock 的净进展（望远镜式差分，无悬停农分） ----------
    dist_prev = ((obs[12] * 5.0) ** 2 + (obs[13] * 4.0) ** 2) ** 0.5
    dist_next = ((next_obs[12] * 5.0) ** 2 + (next_obs[13] * 4.0) ** 2) ** 0.5
    progress = dist_prev - dist_next
    if progress > 0.25:
        progress = 0.25
    elif progress < -0.25:
        progress = -0.25
    crate_progress = 15.0 * progress

    # ---------- 完成度探针：近 dock + 近静止 + 轴向对齐（收紧阈值 × 乘积） ----------
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    distance_factor = max(0.0, 1.0 - dist_next / 0.40)
    speed_factor = max(0.0, 1.0 - crate_speed / 0.15)
    alignment_factor = max(0.0, 1.0 - abs(2.0 * next_obs[10] * next_obs[11]) / 0.8660254)
    docking_settle = 2.0 * distance_factor * speed_factor * alignment_factor

    # ---------- 脆弱货物保护：接触瞬间相对接近速度过大则惩罚 ----------
    contact_penalty = 0.0
    if next_obs[14] > 0.5:
        cart_vx = next_obs[4] * 3.0 * next_obs[2]
        cart_vy = next_obs[4] * 3.0 * next_obs[3]
        crate_vx = next_obs[8] * 3.0
        crate_vy = next_obs[9] * 3.0
        rel_speed = ((cart_vx - crate_vx) ** 2 + (cart_vy - crate_vy) ** 2) ** 0.5
        if rel_speed > 0.25:
            contact_penalty = -2.0 * (rel_speed - 0.25) ** 2

    # ---------- 边界保护：小车与货箱均不得离开仓库地板 ----------
    boundary_penalty = 0.0
    cart_x_ratio = abs(next_obs[0])
    cart_y_ratio = abs(next_obs[1])
    if cart_x_ratio > 0.9:
        boundary_penalty -= 2.0 * (cart_x_ratio - 0.9) ** 2
    if cart_y_ratio > 0.