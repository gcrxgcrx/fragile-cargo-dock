# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 主信号：货箱到泊位的进度（delta，避免悬停陷阱） ----------
    dock_x_old = obs[12]
    dock_y_old = obs[13]
    dock_x_new = next_obs[12]
    dock_y_new = next_obs[13]

    dist_old = (dock_x_old * dock_x_old + dock_y_old * dock_y_old) ** 0.5
    dist_new = (dock_x_new * dock_x_new + dock_y_new * dock_y_new) ** 0.5

    progress = dist_old - dist_new          # >0 表示靠近泊位
    crate_to_dock_progress = 6.0 * progress

    # ---------- 停靠质量：位置 / 朝向 / 速度 联合近似 ----------
    # 位置因子：越接近泊位中心越好（连续 bounded）
    pos_factor = 1.0 / (1.0 + 40.0 * dist_new)

    # 朝向因子：货箱朝向误差（用 cos 近似，对齐时接近 1）
    crate_cos = next_obs[10]
    crate_sin = next_obs[11]
    heading_norm = (crate_cos * crate_cos + crate_sin * crate_sin) ** 0.5
    if heading_norm > 1e-6:
        cos_err = crate_cos / heading_norm
    else:
        cos_err = 0.0
    if cos_err < 0.0:
        cos_err = 0.0
    # 30° 对齐阈值对应 cos≈0.866，用线性衰减门
    orient_factor = (cos_err - 0.5) / 0.5
    if orient_factor < 0.0:
        orient_factor = 0.0
    if orient_factor > 1.0:
        orient_factor = 1.0

    # 速度因子：货箱世界系速度（归一化后 / 3.0 量纲），越慢越好
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + 15.0 * crate_speed)

    # 联合满足（几何平均，避免乘积塌缩）
    joint = (pos_factor * orient_factor * speed_factor) ** (1.0 / 3.0)
    crate_docking_quality = 3.0 * joint

    # ---------- 接近泊位时的速度抑制（仅在货箱接近泊位时启用） ----------
    # 距离越近，速度惩罚越强；用 hinge 让远距离时几乎不罚
    near_gate = 1.0 / (1.0 + 60.0 * dist_new)
    over_speed = crate_speed - 0.05
    if over_speed < 0.0:
        over_speed = 0.0
    crate_speed_penalty_near_dock = -1.5 * near_gate * (over_speed ** 2)

    # ---------- 软接触惩罚（间接推断：接触 + 货箱速度偏高） ----------
    contact = next_obs[14]
    if contact > 0.5:
        # 接触时货箱速度越大，越可能是硬碰撞
        excess = crate_speed - 0.4
        if excess < 0.0:
            excess = 0.0
        soft_contact_penalty = -0.8 * (excess ** 2)
    else:
        soft_contact_penalty = 0.0

    # ---------- 越界惩罚（小车与货箱靠近边界时 hinge） ----------
    # 小车位置：obs[0] ∈ [-2,2]（归一化），仓库半宽/半高映射，安全边界留余量
    cart_x = next_obs[0]
    cart_y = next_obs[1]
    cart_out = 0.0
    ax = abs(cart_x)
    if ax > 0.85:
        cart_out += (ax - 0.85) ** 2
    ay = abs(cart_y)
    if ay > 0.85:
        cart_out += (ay - 0.85) ** 2

    # 货箱到泊位偏移：泊位在仓库内，偏移过大意味着可能越界
    crate_out = 0.0
    cdx = abs(next_obs[12])
    cdy = abs(next_obs[13])
    if cdx > 0.9:
        crate_out += (cdx - 0.9) ** 2
    if cdy > 0.9:
        crate_out += (cdy - 0.9) ** 2

    out_of_bounds_penalty = -3.0 * (cart_out + crate_out)

    # ---------- 动作平滑（轻量，仅抑制极端抖动） ----------
    a0 = action[0]
    a1 = action[1]
    action_smoothness = -0.05 * (a0 * a0 + a1 * a1)

    components = {
        "crate_to_dock_progress": float(crate_to_dock_progress),
        "crate_docking_quality": float(crate_docking_quality),
        "crate_speed_penalty_near_dock": float(crate_speed_penalty_near_dock),
        "soft_contact_penalty": float(soft_contact_penalty),
        "out_of_bounds_penalty": float(out_of_bounds_penalty),
        "action_smoothness": float(action_smoothness),
    }

    total_reward = (
        crate_to_dock_progress
        + crate_docking_quality
        + crate_speed_penalty_near_dock
        + soft_contact_penalty
        + out_of_bounds_penalty
        + action_smoothness
    )

    return float(total_reward), components
```
