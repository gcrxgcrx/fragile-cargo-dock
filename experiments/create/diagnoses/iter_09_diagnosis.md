# Response Record

1. `evidence`：20/20 全部 truncated（len=400），无 terminated，说明策略既未成功也未触发失败终止；`crate_to_dock_progress` 是唯一有实质量级的信号（episode_sum_mean=29.47，signed_share 68.1%，active_rate 39%），`joint_dock_completion` 100% 激活但均值仅 13.72，两个惩罚组件 active_rate 0.1%/0.2% 近乎僵尸；历史 best 停在 iter3 的 3.86，同骨架族已迭代 ≥4 轮未刷新。
2. `behavior_diagnosis`：策略在"把货箱往 dock 方向推"上取得部分进展（progress 有正累积），但从未进入"低速+对齐+持续 10 步"的完成区，属于长时徘徊型失败——货箱被推近后因无刹车继续滑行/震荡，无法停稳，episode 全部耗尽时间。
3. `signal_completeness`：主信号（接近改善）存在且可达，但缺少"停稳"这一关键职责的强引导——现有 joint 因子权重仅 0.05，被 progress 的 40.0 权重完全淹没，agent 没有动机去减速货箱；两个惩罚组件几乎不触发，属僵尸。
4. `selected_level`：Level 3 重建（触发条件：同骨架族连续 ≥4 轮未刷新 best，且预判列连续 ≥3 轮 ❌）。
5. `selected_intervention`：更换主信号框架——把"接近改善"降为辅助，主信号改为**停靠完成度联合代理**（near × slow × align 的几何平均，连续 bounded，带 floor 防塌缩），并加入**货箱速度 hinge 惩罚**作为停稳引导；删除两个僵尸惩罚组件。
6. `falsifiable_hypothesis`：若失败主因是"缺乏停稳激励"而非"无法接近"，则提高 joint 完成度权重、并让货箱低速直接贡献正奖励后，货箱在 dock 附近的低速停留比例应上升，`joint_dock_completion` 的 episode_sum_mean 应显著增大，且 score 应超过历史 best 3.86。
7. `expected_next_round`：`joint_dock_completion` episode_sum_mean 从 13.72 升至 ≥40；`crate_to_dock_progress` 份额下降至 <40%；`crate_speed_hinge` active_rate 从 0 升至 >20%；score 预期 >5；len 仍可能接近 400（若仍未完成）但若成功则出现 terminated。
8. `main_risk`：joint 几何平均若某因子长期趋近 floor，会退化为常数偏置，agent 可能学会"贴着 dock 慢速徘徊"刷 joint 而不真正对齐停稳——需靠 align 因子和速度 hinge 共同约束。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- unpack (仅使用环境声明的 obs 维度) ----------
    dx = obs[12]
    dy = obs[13]
    ndx = next_obs[12]
    ndy = next_obs[13]

    cvx = obs[8]
    cvy = obs[9]
    crate_speed = (cvx * cvx + cvy * cvy) ** 0.5

    crate_cos = obs[10]
    crate_sin = obs[11]

    cart_fwd = obs[4]
    contact = obs[14]

    # ---------- A. MAIN: dock_completion_joint (几何平均, 连续 bounded) ----------
    # 三因子：near / slow / align，每因子连续 bounded，带 floor 防塌缩
    dist_new = (ndx * ndx + ndy * ndy) ** 0.5
    f_near = max(0.0, 1.0 - dist_new / 0.5)
    f_slow = 1.0 / (1.0 + 40.0 * crate_speed)
    f_align = abs(crate_cos)

    f_near_c = max(f_near, 0.05)
    f_slow_c = max(f_slow, 0.05)
    f_align_c = max(f_align, 0.05)
    joint = (f_near_c * f_slow_c * f_align_c) ** (1.0 / 3.0)

    w_joint = 30.0
    r_joint = w_joint * joint

    # ---------- B. AUX: crate_to_dock_progress (改善量, 低权重) ----------
    dist_old = (dx * dx + dy * dy) ** 0.5
    progress = dist_old - dist_new
    w_progress = 8.0
    r_progress = w_progress * progress

    # ---------- C. crate_speed_hinge (停稳引导, 只在货箱过快时罚) ----------
    # 阈值设在成功速度边界(0.05)之上，给安全区留空间
    speed_excess = max(0.0, crate_speed - 0.15)
    w_speed = 6.0
    r_speed = -w_speed * speed_excess

    # ---------- D. fragile_impact_penalty (hinge, 仅接触时) ----------
    rel_speed = abs(cart_fwd - crate_speed)
    impact_excess = max(0.0, rel_speed - 0.25)
    w_impact = 3.0
    r_impact = -w_impact * impact_excess * contact

    # ---------- total ----------
    total_reward = r_joint + r_progress + r_speed + r_impact

    components = {
        "dock_completion_joint": r_joint,
        "crate_to_dock_progress": r_progress,
        "crate_speed_hinge": r_speed,
        "fragile_impact_penalty": r_impact,
    }

    return float(total_reward), components
```
