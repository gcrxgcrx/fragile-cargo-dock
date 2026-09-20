```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 模块级状态（回合边界重置 + 一次性事件 + 连续计数）----
    if not hasattr(compute_reward, "_state"):
        compute_reward._state = {
            "prev_t": -1.0,
            "streak": 0,
            "paid_success": False,
            "paid_enter": False,
            "prev_dist": None,
        }
    st = compute_reward._state

    t = float(next_obs[18])
    if t < st["prev_t"] or t <= 1.0 / 400.0:
        st["streak"] = 0
        st["paid_success"] = False
        st["paid_enter"] = False
        st["prev_dist"] = None
    st["prev_t"] = t

    # ---- 几何量（泊位容差来自环境事实：|dx|<=0.024, |dy|<=0.030 归一化）----
    dx = float(next_obs[12])
    dy = float(next_obs[13])
    dist = (dx * dx + dy * dy) ** 0.5

    # ---- 货箱速度（世界系，m/s）----
    cvx = float(next_obs[8]) * 3.0
    cvy = float(next_obs[9]) * 3.0
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # ---- 朝向误差（货箱朝向 vs 泊位朝向，泊位朝向按 0 弧度对齐处理）----
    chx = float(next_obs[10])
    chy = float(next_obs[11])
    # 货箱朝向角与其归一化向量；对齐误差用 |sin| 近似
    heading_err_sin = chy
    if heading_err_sin < 0.0:
        heading_err_sin = -heading_err_sin
    # 30 度对应 sin(30)=0.5
    align_ok = 1.0 if heading_err_sin < 0.5 else 0.0

    # ---- 完成条件显式推断 ----
    inside = 1.0 if (dx <= 0.024 and dx >= -0.024 and dy <= 0.030 and dy >= -0.030) else 0.0
    slow = 1.0 if crate_speed < 0.05 else 0.0
    complete_now = 1.0 if (inside > 0.5 and align_ok > 0.5 and slow > 0.5) else 0.0

    if complete_now > 0.5:
        st["streak"] += 1
    else:
        st["streak"] = 0

    # ---- 组件 1：货箱向泊位的增量推进（只在更接近时给分）----
    progress = 0.0
    if st["prev_dist"] is not None:
        improvement = st["prev_dist"] - dist
        if improvement > 0.0:
            progress = 40.0 * improvement
    st["prev_dist"] = dist

    # ---- 组件 2：接近泊位时的货箱速度抑制（门控，仅在近泊位区激活）----
    # 门控：dist 越小门越强；远离泊位不惩罚，避免阻碍推进
    near_gate = 1.0 - dist / 0.20
    if near_gate < 0.0:
        near_gate = 0.0
    if near_gate > 1.0:
        near_gate = 1.0
    speed_pen = -0.5 * near_gate * crate_speed

    # ---- 组件 3：朝向对齐 shaping（仅在近泊位时激活）----
    align_bonus = 0.0
    if near_gate > 0.0:
        align_factor = 1.0 - heading_err_sin
        if align_factor < 0.0:
            align_factor = 0.0
        align_bonus = 0.3 * near_gate * align_factor

    # ---- 组件 4：接触轻柔度（唯一教会"接近泊位减速"的信号）----
    crate_along_heading = cvx * float(obs[2]) + cvy * float(obs[3])
    closing = float(obs[4]) * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if float(next_obs[14]) > 0.5 else 0.0
    gentleness = -0.05 * contact * closing

    # ---- 组件 5：越界 hinge 惩罚（小车与货箱边界）----
    oob_pen = 0.0
    cx = float(next_obs[0])
    cy = float(next_obs[1])
    if cx > 0.95:
        oob_pen -= 0.5 * (cx - 0.95)
    if cx < -0.95:
        oob_pen -= 0.5 * (-0.95 - cx)
    if cy > 0.95:
        oob_pen -= 0.5 * (cy - 0.95)
    if cy < -0.95:
        oob_pen -= 0.5 * (-0.95 - cy)

    # ---- 组件 6：动作平滑（轻量，避免压制必要推动）----
    smooth_pen = -0.01 * (float(action[0]) ** 2 + float(action[1]) ** 2)

    # ---- 组件 7：一次性完成事件 ----
    success_event = 0.0
    if st["streak"] >= 10 and not st["paid_success"]:
        st["paid_success"] = True
        success_event = 300.0

    # ---- 组件 8：一次性首次进入泊位奖励 ----
    enter_event = 0.0
    if inside > 0.5 and not st["paid_enter"]:
        st["paid_enter"] = True
        enter_event = 20.0

    total_reward = (
        progress
        + speed_pen
        + align_bonus
        + gentleness
        + oob_pen
        + smooth_pen
        + success_event
        + enter_event
    )

    components = {
        "crate_to_dock_progress": progress,
        "crate_speed_penalty_near_dock": speed_pen,
        "crate_alignment_shaping": align_bonus,
        "soft_contact_penalty": gentleness,
        "out_of_bounds_penalty": oob_pen,
        "action_smoothness": smooth_pen,
        "success_event": success_event,
        "enter_dock_event": enter_event,
    }

    return (float(total_reward), components)
```