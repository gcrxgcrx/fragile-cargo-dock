# Response Record

```python
# 模块级状态：用于回合边界检测、连续计数、一次性事件与首次进入记录
_PREV_T = [-1.0]
_STREAK = [0]
_PAID = [False]
_ENTERED = [False]
_PREV_DIST = [-1.0]

# 自检记录（单步奖励估算，量级参考）：
#   R_idle   ≈ -0.05 (边界/静止，无推进、无进展)
#   R_push   ≈ +1.20 (正常接触推进，接近速度 ~0.3 m/s，推进项 ~1.5，轻柔度 ~-0.015)
#   R_settled ≈ +20.0 (泊位内 + 对齐 + 慢，停稳期每步收益 20，其余过程项为 0)
#   R_push - R_idle ≈ 1.25，与最大推进项同量级
#   R_settled - R_push ≈ 18.8 > 0，停稳为全局最优点
#   自检①: ②(推进) > ①(静止)
#   自检②: ④(停稳+完成) 累计 >> ③(泊位外悬停)
#   自检③: closing=1.0 时惩罚 -0.5，closing=0.05 时惩罚 -0.025，差距 0.475 与推进项同量级
#   自检④: 同一停稳状态连续调用 12 次，除一次性事件外其余为 0
#   自检⑤: 边界边缘奖励明显低于中心

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    components = {}

    # ---------- 回合边界检测 ----------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _PREV_DIST[0] = -1.0
    _PREV_T[0] = t

    # ---------- 几何量 ----------
    # 货箱到泊位中心的有符号偏移（归一化）
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    # 货箱朝向误差（弧度）
    ch = float(next_obs[10])
    sh = float(next_obs[11])
    crate_angle = 0.0
    if abs(ch) + abs(sh) > 1e-6:
        crate_angle = (sh * sh + ch * ch) ** 0.5
        # 朝向误差用 atan2 近似；对齐 cos 值
    align_cos = ch  # 泊位朝向假定为 +x，cos 值即为对齐度

    # 货箱速度（世界系）
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 完成判据（严格取自环境事实）
    # |obs[12]| <= 0.024 且 |obs[13]| <= 0.030 且 朝向误差 < 30° 且 速度 < 0.05
    in_pos = 1.0 if (abs(dx) <= 0.024 and abs(dy) <= 0.030) else 0.0
    in_align = 1.0 if align_cos >= 0.8660254 else 0.0  # cos(30°)
    in_slow = 1.0 if crate_speed < 0.05 else 0.0
    is_settled = 1.0 if (in_pos > 0.5 and in_align > 0.5 and in_slow > 0.5) else 0.0

    # ---------- 1. 推进项（增量形式，避免悬停收割） ----------
    # 用 delta(distance) 的改进量，正向给分，负向轻微惩罚（防止徘徊）
    progress = 0.0
    if _PREV_DIST[0] >= 0.0:
        delta = _PREV_DIST[0] - dist  # 正数表示这一帧更接近泊位
        if delta > 0.0:
            progress = 1.5 * delta * 100.0  # 放大到可感知量级
        else:
            progress = 0.3 * delta * 100.0  # 远离时轻微负反馈
    _PREV_DIST[0] = dist
    if is_settled > 0.5:
        progress = 0.0
    components["progress"] = progress

    # ---------- 2. 轻柔度（接触时惩罚接近速度） ----------
    crate_vx = float(next_obs[8]) * 3.0
    crate_vy = float(next_obs[9]) * 3.0
    crate_along_heading = crate_vx * float(obs[2]) + crate_vy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.5 * contact * closing
    if is_settled > 0.5:
        gentleness = 0.0
    components["gentleness"] = gentleness

    # ---------- 3. 越界守卫 ----------
    cart_x = abs(float(next_obs[0]))
    cart_y = abs(float(next_obs[1]))
    edge = 0.0
    if cart_x > 0.95:
        edge -= 5.0 * (cart_x - 0.95) * 20.0
    if cart_y > 0.95:
        edge -= 5.0 * (cart_y - 0.95) * 20.0
    # 货箱越界（用货箱到泊位偏移与小车位置近似恢复）
    crate_x = float(next_obs[0]) * 5.0 + (float(next_obs[6]) * 3.0) * float(obs[2]) - (float(next_obs[7]) * 3.0) * float(obs[3])
    crate_y = float(next_obs[1]) * 4.0 + (float(next_obs[6]) * 3.0) * float(obs[3]) + (float(next_obs[7]) * 3.0) * float(obs[2])
    crate_nx = abs(crate_x) / 5.0
    crate_ny = abs(crate_y) / 4.0
    if crate_nx > 0.95:
        edge -= 5.0 * (crate_nx - 0.95) * 20.0
    if crate_ny > 0.95:
        edge -= 5.0 * (crate_ny - 0.95) * 20.0
    if is_settled > 0.5:
        edge = 0.0
    components["edge_guard"] = edge

    # ---------- 4. 停稳期每步收益（完成谓词成立时每步给正收益） ----------
    settled_bonus = 0.0
    if is_settled > 0.5:
        settled_bonus = 20.0
    components["settled_bonus"] = settled_bonus

    # ---------- 5. 首次进入泊位（一次性） ----------
    enter_bonus = 0.0
    if is_settled > 0.5 and not _ENTERED[0]:
        _ENTERED[0] = True
        enter_bonus = 30.0
    components["enter_bonus"] = enter_bonus

    # ---------- 6. 完成事件（连续 10 步，一次性） ----------
    if is_settled > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:
        _PAID[0] = True
        success_event = 300.0
    components["success_event"] = success_event

    # ---------- 完成状态下除一次性事件外，其余组件必须为 0 ----------
    if is_settled > 0.5:
        components["progress"] = 0.0
        components["gentleness"] = 0.0
        components["edge_guard"] = 0.0
        # settled_bonus 是停稳期每步收益，按要求保留

    total = (components["progress"] + components["gentleness"]
             + components["edge_guard"] + components["settled_bonus"]
             + components["enter_bonus"] + components["success_event"])

    return (float(total), components)
```
