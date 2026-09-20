```python
# ---- 模块级状态（用于回合边界检测、完成事件一次性发放、停稳连续计数） ----
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]

# ---- 自检记录（估算的单步平均奖励） ----
# R_idle    ≈ -0.30  (什么都不做：无推进增量，仅有微小动作代价/无边界罚)
# R_push    ≈ +1.60  (正常匀速推箱：推进增量 ~1.8，轻柔项 ~0，动作代价 ~-0.2)
# R_settled ≈ +20.0  (泊位内 + 对齐 + 慢：停稳期每步收益 +20，其余组件恰好为 0)
# 满足：R_push > R_idle（差距 1.9 >> 最大罚项量级 ~0.3）；
#       R_settled > R_push（20 >> 1.6）。
#
# 自检 ①：②推箱(有速度) > ①静止(初始位置)，因推进增量项在推进时为正。
# 自检 ②：④真正入坞停稳累计 >> ③泊位外 0.3m 悬停（悬停无增量、无停稳收益、无完成事件）。
# 自检 ③：⑤closing=1.0 时轻柔惩罚 -0.05*1.0*1.0*20=-1.0，与推进增量同量级；
#        ⑥closing=0.05 时惩罚 -0.05，差距 0.95 ≈ 推进项量级。
# 自检 ④：停稳状态连续调用 12 次，除首次进入/完成事件外，停稳期每步 +20 线性增长。
# 自检 ⑤：|obs[0]|=1.05 时边界罚单调下降，明显低于场地中央。

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 回合边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
    _PREV_T[0] = t

    # ---------- 观测解码 ----------
    cart_x = obs[0]
    cart_y = obs[1]
    ch = obs[2]
    sh = obs[3]
    cart_fwd = obs[4]
    yaw = obs[5]

    crate_dx = obs[12]   # 货箱中心到泊位中心有符号 x 偏移 / 仓库半宽
    crate_dy = obs[13]   # 货箱中心到泊位中心有符号 y 偏移 / 仓库半高
    n_dx = next_obs[12]
    n_dy = next_obs[13]

    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5

    crate_cos = obs[10]
    crate_sin = obs[11]
    crate_ang = 0.0
    if crate_cos > 1e-6 or crate_cos < -1e-6:
        crate_ang = crate_sin / (abs(crate_cos) + 1e-6)
    # 朝向误差代理：|sin| 与 |1-cos| 组合，越小越对齐
    align_err = (crate_sin * crate_sin) + (1.0 - crate_cos) * (1.0 - crate_cos)
    align_err = align_err ** 0.5

    contact = 1.0 if next_obs[14] > 0.5 else 0.0

    # ---------- 完成条件（严格取自环境事实的泊位几何与阈值） ----------
    inside = 1.0 if (abs(n_dx) <= 0.024 and abs(n_dy) <= 0.030) else 0.0
    aligned = 1.0 if align_err < 0.5 else 0.0        # 朝向误差 < 30° 的连续代理阈值
    slow = 1.0 if crate_speed < 0.05 else 0.0
    settled = 1.0 if (inside > 0.5 and aligned > 0.5 and slow > 0.5) else 0.0

    if settled > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    components = {}

    # ---------- 1. 完成事件（一次性，整个 episode 只发一次） ----------
    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    # ---------- 2. 停稳期每步收益（泊位内 + 对齐 + 慢） ----------
    # 在完成状态下这是唯一的持久正项；其余组件在完成状态下必须恰好为 0。
    settle_bonus = 0.0
    if settled > 0.5:
        settle_bonus = 20.0
    components["settle_bonus"] = settle_bonus

    # ---------- 3. 首次进入泊位（一次性，整局只发一次） ----------
    enter_bonus = 0.0
    if inside > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 5.0
    components["enter_bonus"] = enter_bonus

    # ---------- 4. 货箱到泊位的推进增量（增量形式，避免悬停收割） ----------
    # 用负距离度量，只在"这一帧更接近"时给分。
    prev_dist = (crate_dx * crate_dx + crate_dy * crate_dy) ** 0.5
    new_dist = (n_dx * n_dx + n_dy * n_dy) ** 0.5
    progress = prev_dist - new_dist
    if progress < 0.0:
        progress = 0.0
    # 完成状态下必须为 0：已入坞停稳时不再给推进增量
    if settled > 0.5:
        progress = 0.0
    progress_reward = 6.0 * progress
    components["crate_progress"] = progress_reward

    # ---------- 5. 接触轻柔度（唯一能教会策略减速的信号） ----------
    crate_along_heading = crate_vx * ch + crate_vy * sh
    closing = cart_fwd * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    gentleness = 0.0
    if settled <= 0.5:
        gentleness = -0.05 * contact * closing * 20.0   # k 放大到与推进项同量级
    components["gentleness"] = gentleness

    # ---------- 6. 越界守卫（小车与货箱） ----------
    oob = 0.0
    ax = abs(cart_x)
    ay = abs(cart_y)
    if ax > 0.95:
        oob -= 8.0 * (ax - 0.95)
    if ay > 0.95:
        oob -= 8.0 * (ay - 0.95)
    # 货箱越界代理：货箱相对小车位置恢复到世界坐标（近似）
    crate_wx = cart_x + (obs[6] * 3.0) * ch - (obs[7] * 3.0) * sh
    crate_wy = cart_y + (obs[6] * 3.0) * sh + (obs[7] * 3.0) * ch
    cax = abs(crate_wx)
    cay = abs(crate_wy)
    if cax > 0.95:
        oob -= 8.0 * (cax - 0.95)
    if cay > 0.95:
        oob -= 8.0 * (cay - 0.95)
    components["out_of_bounds"] = oob

    # ---------- 7. 动作代价（轻量，不压制推进） ----------
    act_cost = 0.0
    if settled <= 0.5:
        act_cost = -0.05 * (action[0] * action[0] + action[1] * action[1])
    components["action_cost"] = act_cost

    # ---------- 完成状态下，除一次性事件外其余组件必须恰好为 0 ----------
    if settled > 0.5:
        # 保留 success_event 与 settle_bonus；其余置零
        components["enter_bonus"] = 0.0
        components["crate_progress"] = 0.0
        components["gentleness"] = 0.0
        components["out_of_bounds"] = 0.0
        components["action_cost"] = 0.0

    total = 0.0
    for k in components:
        total += components[k]

    return (float(total), components)
```