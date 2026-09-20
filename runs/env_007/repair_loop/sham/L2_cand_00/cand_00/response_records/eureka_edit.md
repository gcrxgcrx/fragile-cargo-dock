# Response Record

分析：主进度信号 `crate_to_dock_progress` 激活率几乎为 0（0.0%–0.7%），因为 `progress = d_now - d_nxt` 只有在单步净接近时才为正，被 `near_gate` 和 `gate` 进一步压缩，导致货箱几乎从未被推动——20 个 episode 全部 truncation、无成功。`soft_contact`、`dock_speed_penalty`、`bounds_guard` 等惩罚项在训练后期几乎不激活，说明策略学会了"静止不动"以规避惩罚。`enter_dock_event`/`success_event` 偶发负值说明状态重置逻辑与 episode 边界不同步。改进方向：把主进度改成对"货箱到泊位距离"的**凸化稠密信号**（而非稀疏 delta），用几何平均的联合完成度 proxy 提供连续梯度，把速度/接触惩罚做成**门控**而非独立扣分，并确保"推箱靠近泊位"的单步奖励严格高于"静止不动"。

```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 几何量 ----------
    dx_now = obs[12]
    dy_now = obs[13]
    dx_nxt = next_obs[12]
    dy_nxt = next_obs[13]

    d_now = (dx_now * dx_now + dy_now * dy_now) ** 0.5
    d_nxt = (dx_nxt * dx_nxt + dy_nxt * dy_nxt) ** 0.5

    TOL_X = 0.024
    TOL_Y = 0.030
    inside = 1.0 if (abs(dx_nxt) <= TOL_X and abs(dy_nxt) <= TOL_Y) else 0.0

    # 朝向对齐：|sin| 越小越对齐
    c_cos = next_obs[10]
    c_sin = next_obs[11]
    align = 1.0 - abs(c_sin)
    if align < 0.0:
        align = 0.0
    if align > 1.0:
        align = 1.0
    aligned_ok = 1.0 if abs(c_sin) <= 0.5 else 0.0

    # 货箱速度
    cvx = next_obs[8] * 3.0
    cvy = next_obs[9] * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5
    slow_ok = 1.0 if crate_speed < 0.05 else 0.0

    done_cond = 1.0 if (inside > 0.5 and aligned_ok > 0.5 and slow_ok > 0.5) else 0.0

    if done_cond > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    components = {}

    # ---------- 1) 主进度：凸化稠密接近信号 + delta 改进 ----------
    # 稠密接近：距离越小分越高（凸化），保证每步都有梯度
    prox = 1.0 / (1.0 + 6.0 * d_nxt)
    # delta 改进：单步净接近（仅正向）
    progress = d_now - d_nxt
    if progress < 0.0:
        progress = 0.0
    # 对齐门控：对齐越好，接近信号越有价值（0.4~1.0，不归零）
    gate = 0.4 + 0.6 * align
    components["crate_to_dock_progress"] = (2.0 * prox + 45.0 * progress) * gate

    # ---------- 2) 联合完成度 proxy（几何平均，连续有梯度） ----------
    f_pos = 1.0 / (1.0 + 40.0 * d_nxt)          # 位置因子
    f_align = align                              # 朝向因子
    f_slow = 1.0 / (1.0 + 20.0 * crate_speed)    # 静止因子
    joint = (f_pos * f_align * f_slow) ** (1.0 / 3.0)
    components["docking_quality"] = 6.0 * joint

    # ---------- 3) 轻柔接触（门控式，仅在接近时惩罚过大闭合速度） ----------
    crate_along = cvx * next_obs[2] + cvy * next_obs[3]
    closing = obs[4] * 3.0 - crate_along
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    # 只在接近泊位、且闭合速度大时才惩罚；远离泊位时推动不受罚
    near_dock_gate = 1.0 / (1.0 + 6.0 * d_nxt)
    over_close = closing - 0.8
    if over_close < 0.0:
        over_close = 0.0
    components["soft_contact"] = -1.5 * contact * near_dock_gate * over_close

    # ---------- 4) 泊位内速度抑制（仅在已进入泊位时惩罚超速） ----------
    over_speed = crate_speed - 0.05
    if over_speed < 0.0:
        over_speed = 0.0
    components["dock_speed_penalty"] = -4.0 * inside * over_speed

    # ---------- 5) 越界守卫 ----------
    ax = abs(next_obs[0])
    ay = abs(next_obs[1])
    over_x = ax - 0.95
    if over_x < 0.0:
        over_x = 0.0
    over_y = ay - 0.95
    if over_y < 0.0:
        over_y = 0.0
    components["bounds_guard"] = -80.0 * (over_x * over_x + over_y * over_y) - 20.0 * (over_x + over_y)

    # ---------- 6) 障碍接近惩罚 ----------
    sf = next_obs[15]
    sl = next_obs[16]
    sr = next_obs[17]
    obs_pen = 0.0
    if sf > 0.9:
        obs_pen += (sf - 0.9)
    if sl > 0.95:
        obs_pen += (sl - 0.95)
    if sr > 0.95:
        obs_pen += (sr - 0.95)
    components["obstacle_penalty"] = -5.0 * obs_pen

    # ---------- 7) 首次进入泊位事件 ----------
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        components["enter_dock_event"] = 20.0
    else:
        components["enter_dock_event"] = 0.0

    # ---------- 8) 完成事件（连续 10 步满足） ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 250.0
    components["success_event"] = success_event

    total = 0.0
    for k in components:
        total += components[k]

    return float(total), components
```
