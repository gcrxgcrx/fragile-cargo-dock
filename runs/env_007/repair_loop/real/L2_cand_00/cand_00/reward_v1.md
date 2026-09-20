分析：任务分数 -1.23 且 20/20 全部 truncation，说明策略从未成功停靠；`crate_to_dock_progress` 每步几乎为 0（激活率 <1%），主信号太稀疏——`progress` 用了 `max(0, d_now-d_nxt)` 的差分，在货箱几乎不动时恒为 0，无梯度。`bounds_guard` 在 10% 时高达 -7.66 且 max 达 -137，说明早期它主导了总奖励，把策略压成"不敢动"，之后策略学会了"贴着边界不动"（激活率降到 0 但任务分数没升）。`enter_dock_event`/`success_event` 全程 0，一次性稀疏奖励对探索无帮助。改法：主进度改为**基于势能的稠密信号**（用 `-d_nxt` 的势能差加常数偏置，保证"接近泊位"持续给正分而非只在位移时给分），并把边界守卫从"独立大惩罚"改为**软门控**乘到主进度上，避免早期压制探索；同时用几何平均的联合完成代理替代纯二值事件，提供连续梯度。

```python
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- episode 边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 几何量（全部来自已声明维度） ----------
    dx_now = obs[12]
    dy_now = obs[13]
    dx_nxt = next_obs[12]
    dy_nxt = next_obs[13]

    d_now = (dx_now * dx_now + dy_now * dy_now) ** 0.5
    d_nxt = (dx_nxt * dx_nxt + dy_nxt * dy_nxt) ** 0.5

    TOL_X = 0.024
    TOL_Y = 0.030
    inside = 1.0 if (abs(dx_nxt) <= TOL_X and abs(dy_nxt) <= TOL_Y) else 0.0

    # 货箱朝向：|sin| 越小越对齐；cos(30°)=0.866 -> |sin|<=0.5 视为对齐
    c_cos = next_obs[10]
    c_sin = next_obs[11]
    align = 1.0 - abs(c_sin)
    if align < 0.0:
        align = 0.0
    if align > 1.0:
        align = 1.0
    aligned_ok = 1.0 if abs(c_sin) <= 0.5 else 0.0

    # 货箱速度（m/s）
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

    # ---------- 越界软门控（替代独立大惩罚） ----------
    # 用小车位置构造连续门：接近 ±1.0 时从 1 衰减到 0
    ax = abs(next_obs[0])
    ay = abs(next_obs[1])
    gx = (1.0 - ax) / 0.15
    if gx < 0.0:
        gx = 0.0
    if gx > 1.0:
        gx = 1.0
    gy = (1.0 - ay) / 0.15
    if gy < 0.0:
        gy = 0.0
    if gy > 1.0:
        gy = 1.0
    bounds_gate = gx * gy

    # ---------- 1) 主进度：稠密势能信号（保证接近泊位持续给正分） ----------
    # 用 bounded 距离的势能：越接近泊位，基础分越高；同时叠加位移改进
    # 势能项：1/(1+5*d) 在 d=0 时为 1，d=0.5 时约 0.29
    pot_now = 1.0 / (1.0 + 5.0 * d_now)
    pot_nxt = 1.0 / (1.0 + 5.0 * d_nxt)
    # 位移改进（正向才给分，避免震荡刷分）
    delta = pot_nxt - pot_now
    if delta < 0.0:
        delta = 0.0
    # 对齐门：对齐越好，接近泊位的价值越高
    align_gate = 0.4 + 0.6 * align
    # 主信号 = (势能基线 + 位移改进) * 对齐门 * 边界门
    progress_signal = (0.35 * pot_nxt + 3.0 * delta) * align_gate * bounds_gate
    components["crate_to_dock_progress"] = 4.0 * progress_signal

    # ---------- 2) 联合完成代理（连续几何平均，提供稠密梯度） ----------
    f_inside = 1.0 / (1.0 + 20.0 * d_nxt)
    f_align = align
    f_slow = 1.0 / (1.0 + 20.0 * crate_speed)
    joint = (f_inside * f_align * f_slow) ** (1.0 / 3.0)
    components["docking_quality"] = 2.0 * joint * bounds_gate

    # ---------- 3) 轻柔接触（只在接触且正在高速逼近时惩罚） ----------
    crate_along_heading = cvx * next_obs[2] + cvy * next_obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    # hinge：只有 closing 超过 0.3 m/s 才罚，避免误伤正常推动
    over_close = closing - 0.3
    if over_close < 0.0:
        over_close = 0.0
    components["soft_contact"] = -1.5 * contact * over_close

    # ---------- 4) 接近泊位时的速度抑制（hinge，只在速度超阈值时罚） ----------
    near_dock = 1.0 / (1.0 + 8.0 * d_nxt)
    over_speed = crate_speed - 0.08
    if over_speed < 0.0:
        over_speed = 0.0
    components["dock_speed_penalty"] = -2.5 * near_dock * over_speed

    # ---------- 5) 障碍接近惩罚（hinge，只在接近接触时） ----------
    sf = next_obs[15]
    sl = next_obs[16]
    sr = next_obs[17]
    obs_pen = 0.0
    if sf > 0.85:
        obs_pen += (sf - 0.85)
    if sl > 0.9:
        obs_pen += (sl - 0.9)
    if sr > 0.9:
        obs_pen += (sr - 0.9)
    components["obstacle_penalty"] = -6.0 * obs_pen

    # ---------- 6) 一次性：首次进入泊位 ----------
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        components["enter_dock_event"] = 20.0
    else:
        components["enter_dock_event"] = 0.0

    # ---------- 7) 一次性：完成事件（连续 10 步满足完成条件） ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 200.0
    components["success_event"] = success_event

    # ---------- 8) 完成状态下其余组件归零 ----------
    if done_cond > 0.5:
        components["crate_to_dock_progress"] = 0.0
        components["docking_quality"] = 0.0
        components["soft_contact"] = 0.0
        components["dock_speed_penalty"] = 0.0
        components["obstacle_penalty"] = 0.0

    total = 0.0
    for k in components:
        total += components[k]

    return float(total), components
```