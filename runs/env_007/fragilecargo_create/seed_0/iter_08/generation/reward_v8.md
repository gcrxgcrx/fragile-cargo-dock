`evidence`: 7 轮全部 truncated 为主（19/20），score 从未超过 3.86，本轮 -6.06；`crate_to_dock_progress`、`fragile_impact_penalty`、`local_obstacle_penalty` 三个组件 active_rate=0%、episode_sum=0，唯一有信号的 `joint_dock_completion` 是纯状态占据型 proxy（56.65 全为正），与外部 score 反向。

`behavior_diagnosis`: 策略发现"停在货箱附近保持对齐"即可持续领取 joint proxy 奖励，从不真正推动货箱向 dock 移动——典型 proxy 徘徊 exploit；三个真正与任务进展相关的组件全部零激活，说明它们要么被 proxy 淹没，要么信号形态无法产生梯度。

`signal_completeness`: 主进展职责（货箱向 dock 位移）在代码中存在但完全失效（delta 恒为 0，说明货箱从未被推动）；完成职责被状态占据型 proxy 冒充；易碎/边界约束从未触发。信号名义齐全但主信号不可达。

`selected_level`: Level 3 重建（累积记录同骨架族 ≥4 轮未刷新 best=3.86，且 Level 2 形态变换后得分持续恶化）。

`selected_intervention`: 更换主信号框架——废弃"状态占据型 joint proxy"作为主信号，改为 **improvement_delta 主信号（货箱到 dock 距离的逐步减少）+ 联合完成 proxy 仅作低权重辅助 + 易碎冲击 hinge 惩罚**，并把完成 proxy 从"状态值"改为"改善量"，消除停留刷分。

`falsifiable_hypothesis`: 若 proxy 徘徊是主因，则把主信号换成"距离减少量"后，`crate_to_dock_progress` 的 active_rate 应显著 >0（货箱被推动），且 `joint_dock_completion` 的 episode_sum 应大幅下降（不再靠停留刷分），score 应上升。

`expected_next_round`: `crate_to_dock_progress` active_rate 从 0% 升至 >30%，episode_sum_mean 由 0 变为非零；`joint_dock_completion` episode_sum_mean 从 56.65 降至 <10；score 从 -6.06 升至 >0；len 保持 ~400（不因新惩罚暴跌）。

`main_risk`: 若货箱初始就贴近 dock，delta 主信号会迅速饱和导致后期无梯度；或冲击 hinge 阈值过严压制接触推动。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- unpack (仅使用环境声明的 obs 维度) ----------
    # 货箱到 dock 的有符号偏移（归一化）
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    # 货箱世界速度
    cvx = obs[8]
    cvy = obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    # 货箱朝向
    crate_cos = obs[10]
    crate_sin = obs[11]

    # 小车前向速度、接触
    cart_fwd = obs[4]
    contact = obs[14]

    # 静态障碍接近度
    s_front = obs[15]
    s_left = obs[16]
    s_right = obs[17]

    # ---------- A. MAIN: crate_to_dock_progress (improvement_delta) ----------
    # 主信号 = 货箱到 dock 距离的逐步减少量（改善量，非状态值）
    dist_old = (dx * dx + dy * dy) ** 0.5
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    progress = dist_old - dist_new  # >0 表示货箱确实向 dock 靠近

    w_progress = 40.0
    r_progress = w_progress * progress

    # ---------- B. AUX: joint_dock_completion (低权重，几何平均，防塌缩) ----------
    # 三因子：near / slow / align，每因子连续 bounded，带 floor
    f_near = max(0.0, 1.0 - dist_new / 0.5)
    f_slow = 1.0 / (1.0 + 40.0 * crate_speed)
    f_align = abs(crate_cos)

    f_near_c = max(f_near, 0.05)
    f_slow_c = max(f_slow, 0.05)
    f_align_c = max(f_align, 0.05)
    joint = (f_near_c * f_slow_c * f_align_c) ** (1.0 / 3.0)

    # 辅助权重显著压低，避免状态占据刷分
    w_joint = 0.05
    r_joint = w_joint * joint

    # ---------- C. fragile_impact_penalty (hinge, 仅接触时) ----------
    # 相对速度超过阈值才罚，阈值设在易碎冲击边界附近
    rel_speed = abs(cart_fwd - crate_speed)
    impact_excess = max(0.0, rel_speed - 0.25)
    w_impact = 3.0
    r_impact = -w_impact * impact_excess * contact

    # ---------- D. local_obstacle_penalty (hinge, 局部障碍接近度) ----------
    obs_excess = max(0.0, s_front - 0.8) + max(0.0, s_left - 0.8) + max(0.0, s_right - 0.8)
    w_obs = 2.0
    r_obs = -w_obs * obs_excess

    # ---------- total ----------
    total_reward = r_progress + r_joint + r_impact + r_obs

    components = {
        "crate_to_dock_progress": r_progress,
        "joint_dock_completion": r_joint,
        "fragile_impact_penalty": r_impact,
        "local_obstacle_penalty": r_obs,
    }

    return float(total_reward), components
```