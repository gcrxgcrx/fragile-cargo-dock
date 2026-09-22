# ⚠️ 上一版代码验证失败
错误信息：Reward v9 failed validation: 代码无法解析 AST: '(' was never closed (<unknown>, line 168) (record: runs\env_007\fragilecargo_create_v9\seed_0\iter_09\generation\validations\reward_v9.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
## 诊断

1. `evidence`：score=309.55 已是历史 best（=target 250 的 123.8%），terminated=20/20、len=180.9，score_range 仅 [309.12, 310.57]；`terminal_success` episode_sum_mean=300.0（signed 95.5%，每回合恰好触发一次），`settle`=+2.59（0.8%），`hard_hit`/`terminal_failure` active=0%；iter 7 放宽 settle 门（0.45/0.10）→294.21，iter 8 还原（0.35/0.08）→309.55。
2. `behavior_diagnosis`：策略已稳定 100% 入坞成功，无徘徊、无刷分、无早失败（early_terminal=0/20）；唯一残留是终局 settle 相位 `Phi_end≈0.86<1`，有约 0.4–0.9 的正向整形未被计入。
3. `signal_completeness`：9 项作者职责全部存在且校准正确；19 维观测无未用且可解释终止模式者；作者权重被结构块锁死，唯一自由杠杆是自加的势函数 `settle`，且它被低估。
4. `selected_level`：**Level 1（尺度修复）**。iter 6/7/8 未刷新 best 并非方向错误，而是任务已到几何上限（progress≈初始货箱-坞距、approach≈初始车-箱距、dock_enter=5、terminal_success=300 均已封顶），**不触发 Level 3**——重建一个已解且锁权重的骨架只会破坏成功。
5. `selected_intervention`：仅改 `settle` 系数 3.0 → 4.0（不动 gate/ramp，iter 7/8 已证明现带宽更优）。
6. `falsifiable_hypothesis`：`settle` 是精确势函数整形（γ=1、望远镜式），缩放它不改变最优策略，只把净回报提高 (4.0−3.0)·(Φ_end−Φ_start)≈+0.86，因此 score 应上升而成功率保持 20/20。若 score 未升 ≥0.3，或 len 变长/成功下降，则假设被证伪。
7. `expected_next_round`：score 309.55→约 310.3–310.8；`settle` episode_sum_mean 2.59→约 3.4；terminated 20/20、len 约 180–182；`hard_hit`/`terminal_failure` 仍 0。
8. `main_risk`：放大整形系数可能扰动早期训练梯度（在函数逼近下并非严格策略不变），导致轨迹偏移、episode 变长或成功裕度下降；且增益落在评测噪声带 [309.12, 310.57] 内可能不可测。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Faithful reconstruction of the author reward structure.
    # 9 author terms (exact weights) + one policy-invariant
    # potential-based settle term.
    #
    # THIS ROUND (iter 9): LEVEL-1 SCALE FIX on the ONLY free
    #   component, `settle`.
    #   settle coefficient 3.0 -> 4.0.
    #   Rationale: the author weights are frozen by the structure
    #   block and every author term is already at its geometric
    #   maximum (progress ~= initial crate->dock distance,
    #   approach ~= initial cart->crate distance, dock_enter one-shot,
    #   terminal_success one-shot). The sole residual is that the
    #   settle potential saturates at Phi_end ~= 0.86 < 1, so the
    #   dense terminal-rest/aligned/in-dock signal is underweighted.
    #   Because settle is EXACT potential-based shaping (gamma = 1),
    #   scaling it is policy-invariant and only rescales net return.
    #   The gate band (0.35 / 0.08) and all ramps are UNCHANGED:
    #   iter 7 (widened band) -> 294.21 ; iter 8 (reverted) -> 309.55,
    #   so the current band is kept.
    # ============================================================

    # ---------- module-level episode state (no self / helper / info) ----------
    st = getattr(compute_reward, "_r9_state", None)
    if st is None:
        st = {
            "prev_cc": None,      # previous cart->crate distance (m)
            "prev_cd": None,      # previous crate->dock distance (m)
            "prev_t": None,       # previous time_fraction (episode boundary detector)
            "entered": False,     # crate ever fully inside dock tolerance
            "stable": 0,          # consecutive settled steps inside dock
            "hard": 0,            # accumulated hard-impact proxy count
            "succ_paid": False,
            "fail_paid": False,
        }
        compute_reward._r9_state = st

    # ---------- episode boundary detection ----------
    # time_fraction is monotone inside one episode; a drop means a new episode.
    t_now = float(obs[18])
    if st["prev_t"] is not None and t_now < st["prev_t"] - 0.05:
        st["prev_cc"] = None
        st["prev_cd"] = None
        st["entered"] = False
        st["stable"] = 0
        st["hard"] = 0
        st["succ_paid"] = False
        st["fail_paid"] = False
    st["prev_t"] = t_now

    # ---------- geometry (all restored to metres) ----------
    hx = float(obs[2])
    hy = float(obs[3])

    rel_x = float(obs[6]) * 3.0          # cart->crate, body frame x (m)
    rel_y = float(obs[7]) * 3.0          # cart->crate, body frame y (m)
    nrel_x = float(next_obs[6]) * 3.0
    nrel_y = float(next_obs[7]) * 3.0
    cc_next = (nrel_x * nrel_x + nrel_y * nrel_y) ** 0.5      # cart->crate dist (m)

    dnx = float(next_obs[12]) * 5.0      # crate->dock signed x (m)
    dny = float(next_obs[13]) * 4.0      # crate->dock signed y (m)
    cd_next = (dnx * dnx + dny * dny) ** 0.5                  # crate->dock dist (m)

    # ---------- approach_cargo : +1.0 / m (signed potential difference) ----------
    if st["prev_cc"] is None:
        st["prev_cc"] = cc_next
    approach_cargo = st["prev_cc"] - cc_next
    st["prev_cc"] = cc_next

    # ---------- progress : +1.0 / m (signed potential difference) ----------
    if st["prev_cd"] is None:
        st["prev_cd"] = cd_next
    progress = st["prev_cd"] - cd_next
    st["prev_cd"] = cd_next

    # ---------- dock_enter : +5.0 one-shot on first full containment ----------
    inside = (abs(float(next_obs[12])) <= 0.024) and (abs(float(next_obs[13])) <= 0.030)
    dock_enter = 0.0
    if inside and not st["entered"]:
        dock_enter = 5.0
        st["entered"] = True

    # ---------- roughness (weak) + hard_hit (sparse) ----------
    # No impulse channel in obs -> closing-speed x contact proxy.
    # Deliberately sub-dominant so the policy is never afraid to push.
    cart_speed = float(obs[4]) * 3.0
    crate_along = float(obs[8]) * 3.0 * hx + float(obs[9]) * 3.0 * hy
    closing = cart_speed - crate_along
    contact = float(obs[14]) > 0.5

    roughness = 0.0
    hard_hit = 0.0
    if contact and closing > 0.0:
        roughness = -0.02 * closing           # scale kept << progress
        if closing > 1.5:                     # high-severity proxy
            hard_hit = -0.5
            st["hard"] += 1

    # ---------- action_cost / time_cost ----------
    action_cost = -0.0005 * (float(action[0]) ** 2 + float(action[1]) ** 2)
    time_cost = -0.002

    # ---------- terminal_success chain: inside + aligned + slow, 10 consecutive ----------
    crate_speed = ((float(next_obs[8]) * 3.0) ** 2 + (float(next_obs[9]) * 3.0) ** 2) ** 0.5
    aligned = float(next_obs[10]) > 0.866     # |crate heading error| < 30 deg
    if inside and aligned and crate_speed < 0.05:
        st["stable"] += 1
    else:
        st["stable"] = 0

    terminal_success = 0.0
    if st["stable"] >= 10 and not st["succ_paid"]:
        terminal_success = 300.0
        st["succ_paid"] = True

    # ---------- terminal_failure chain: OOB or >=3 hard hits ----------
    crate_wx = float(obs[0]) * 5.0 + rel_x * hx - rel_y * hy
    crate_wy = float(obs[1]) * 4.0 + rel_x * hy + rel_y * hx
    out_of_bounds = (
        abs(float(obs[0])) > 1.05
        or abs(float(obs[1])) > 1.05
        or abs(crate_wx) > 5.25
        or abs(crate_wy) > 4.2
    )

    terminal_failure = 0.0
    if (out_of_bounds or st["hard"] >= 3) and not st["fail_paid"]:
        terminal_failure = -100.0
        st["fail_paid"] = True

    # ---------- settle : potential-based shaping (policy-invariant) ----------
    # Phi = gate(crate->dock dist) * align * ground-speed factor, all in [0,1].
    # reward = 4.0 * (Phi(next) - Phi(prev)) -> telescopes, net-zero on excursions.
    # THIS ROUND: coefficient 3.0 -> 4.0 only (gate band + ramps unchanged).
    dpx = float(obs[12]) * 5.0
    dpy = float(obs[13]) * 4.0
    d_prev = (dpx * dpx + dpy * dpy) ** 0.5

    gate_prev = (0.35 - d_prev) / 0.08
    if gate_prev < 0.0:
        gate_prev = 0.0
    if gate_prev > 1.0:
        gate_prev = 1.0

    align_prev = (float(obs[10]) - 0.70) / 0.30
    if align_prev < 0.0:
        align_prev = 0.0
    if align_prev > 1.0:
        align_prev = 1.0

    spd_prev = ((float(obs[8]) * 3.0) ** 2 + (float(obs[9]) * 3.0) ** 2) ** 0.5
    sfactor_prev = (0.30 - spd_prev) / 0.30
    if sfactor_prev < 0.0:
        sfactor_prev = 0.0
    if sfactor_prev > 1.0:
        sfactor_prev = 1.0

    phi_prev = gate_prev * align_prev * sfactor_prev

    gate_next = (0.35 - cd_next) / 0.08
    if gate_next < 0.0:
        gate_next = 0.0
    if gate_next > 1.0:
        gate_next = 1.0

    align_next = (float

# 1. Search objective
- target_score: 250.000000
- current_score: 309.545139
- gap_to_target: -59.545139
- target_achievement_ratio: 123.818%

# 2. 上一轮奖励函数代码（该轮得分: 309.545139）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # Faithful reconstruction of the author reward structure.
    # 9 author terms (exact weights) + one policy-invariant
    # potential-based settle term.
    #
    # THIS ROUND (iter 8): REVERTED last round's settle gate widening.
    #   gate outer edge 0.45 -> 0.35 m, ramp width 0.10 -> 0.08 m.
    #   Rationale: the widened band (iter 7) coincided with the score
    #   dropping 309.55 -> 294.21; restoring the narrower band re-concentrates
    #   the deceleration/alignment gradient in the immediate docking band.
    #   No author weight is touched.
    # ============================================================

    # ---------- module-level episode state (no self / helper / info) ----------
    st = getattr(compute_reward, "_r8_state", None)
    if st is None:
        st = {
            "prev_cc": None,      # previous cart->crate distance (m)
            "prev_cd": None,      # previous crate->dock distance (m)
            "prev_t": None,       # previous time_fraction (episode boundary detector)
            "entered": False,     # crate ever fully inside dock tolerance
            "stable": 0,          # consecutive settled steps inside dock
            "hard": 0,            # accumulated hard-impact proxy count
            "succ_paid": False,
            "fail_paid": False,
        }
        compute_reward._r8_state = st

    # ---------- episode boundary detection ----------
    # time_fraction is monotone inside one episode; a drop means a new episode.
    t_now = float(obs[18])
    if st["prev_t"] is not None and t_now < st["prev_t"] - 0.05:
        st["prev_cc"] = None
        st["prev_cd"] = None
        st["entered"] = False
        st["stable"] = 0
        st["hard"] = 0
        st["succ_paid"] = False
        st["fail_paid"] = False
    st["prev_t"] = t_now

    # ---------- geometry (all restored to metres) ----------
    hx = float(obs[2])
    hy = float(obs[3])

    rel_x = float(obs[6]) * 3.0          # cart->crate, body frame x (m)
    rel_y = float(obs[7]) * 3.0          # cart->crate, body frame y (m)
    nrel_x = float(next_obs[6]) * 3.0
    nrel_y = float(next_obs[7]) * 3.0
    cc_next = (nrel_x * nrel_x + nrel_y * nrel_y) ** 0.5      # cart->crate dist (m)

    dnx = float(next_obs[12]) * 5.0      # crate->dock signed x (m)
    dny = float(next_obs[13]) * 4.0      # crate->dock signed y (m)
    cd_next = (dnx * dnx + dny * dny) ** 0.5                  # crate->dock dist (m)

    # ---------- approach_cargo : +1.0 / m (signed potential difference) ----------
    if st["prev_cc"] is None:
        st["prev_cc"] = cc_next
    approach_cargo = st["prev_cc"] - cc_next
    st["prev_cc"] = cc_next

    # ---------- progress : +1.0 / m (signed potential difference) ----------
    if st["prev_cd"] is None:
        st["prev_cd"] = cd_next
    progress = st["prev_cd"] - cd_next
    st["prev_cd"] = cd_next

    # ---------- dock_enter : +5.0 one-shot on first full containment ----------
    inside = (abs(float(next_obs[12])) <= 0.024) and (abs(float(next_obs[13])) <= 0.030)
    dock_enter = 0.0
    if inside and not st["entered"]:
        dock_enter = 5.0
        st["entered"] = True

    # ---------- roughness (weak) + hard_hit (sparse) ----------
    # No impulse channel in obs -> closing-speed x contact proxy.
    # Deliberately sub-dominant so the policy is never afraid to push.
    cart_speed = float(obs[4]) * 3.0
    crate_along = float(obs[8]) * 3.0 * hx + float(obs[9]) * 3.0 * hy
    closing = cart_speed - crate_along
    contact = float(obs[14]) > 0.5

    roughness = 0.0
    hard_hit = 0.0
    if contact and closing > 0.0:
        roughness = -0.02 * closing           # scale kept << progress
        if closing > 1.5:                     # high-severity proxy
            hard_hit = -0.5
            st["hard"] += 1

    # ---------- action_cost / time_cost ----------
    action_cost = -0.0005 * (float(action[0]) ** 2 + float(action[1]) ** 2)
    time_cost = -0.002

    # ---------- terminal_success chain: inside + aligned + slow, 10 consecutive ----------
    crate_speed = ((float(next_obs[8]) * 3.0) ** 2 + (float(next_obs[9]) * 3.0) ** 2) ** 0.5
    aligned = float(next_obs[10]) > 0.866     # |crate heading error| < 30 deg
    if inside and aligned and crate_speed < 0.05:
        st["stable"] += 1
    else:
        st["stable"] = 0

    terminal_success = 0.0
    if st["stable"] >= 10 and not st["succ_paid"]:
        terminal_success = 300.0
        st["succ_paid"] = True

    # ---------- terminal_failure chain: OOB or >=3 hard hits ----------
    crate_wx = float(obs[0]) * 5.0 + rel_x * hx - rel_y * hy
    crate_wy = float(obs[1]) * 4.0 + rel_x * hy + rel_y * hx
    out_of_bounds = (
        abs(float(obs[0])) > 1.05
        or abs(float(obs[1])) > 1.05
        or abs(crate_wx) > 5.25
        or abs(crate_wy) > 4.2
    )

    terminal_failure = 0.0
    if (out_of_bounds or st["hard"] >= 3) and not st["fail_paid"]:
        terminal_failure = -100.0
        st["fail_paid"] = True

    # ---------- settle : potential-based shaping (policy-invariant) ----------
    # Phi = gate(crate->dock dist) * align * ground-speed factor, all in [0,1].
    # reward = 3.0 * (Phi(next) - Phi(prev)) -> telescopes, net-zero on excursions.
    # REVERTED THIS ROUND: gate outer edge 0.45 -> 0.35 m, ramp 0.10 -> 0.08 m.
    dpx = float(obs[12]) * 5.0
    dpy = float(obs[13]) * 4.0
    d_prev = (dpx * dpx + dpy * dpy) ** 0.5

    gate_prev = (0.35 - d_prev) / 0.08
    if gate_prev < 0.0:
        gate_prev = 0.0
    if gate_prev > 1.0:
        gate_prev = 1.0

    align_prev = (float(obs[10]) - 0.70) / 0.30
    if align_prev < 0.0:
        align_prev = 0.0
    if align_prev > 1.0:
        align_prev = 1.0

    spd_prev = ((float(obs[8]) * 3.0) ** 2 + (float(obs[9]) * 3.0) ** 2) ** 0.5
    sfactor_prev = (0.30 - spd_prev) / 0.30
    if sfactor_prev < 0.0:
        sfactor_prev = 0.0
    if sfactor_prev > 1.0:
        sfactor_prev = 1.0

    phi_prev = gate_prev * align_prev * sfactor_prev

    gate_next = (0.35 - cd_next) / 0.08
    if gate_next < 0.0:
        gate_next = 0.0
    if gate_next > 1.0:
        gate_next = 1.0

    align_next = (float(next_obs[10]) - 0.70) / 0.30
    if align_next < 0.0:
        align_next = 0.0
    if align_next > 1.0:
        align_next = 1.0

    sfactor_next = (0.30 - crate_speed) / 0.30
    if sfactor_next < 0.0:
        sfactor_next = 0.0
    if sfactor_next > 1.0:
        sfactor_next = 1.0

    phi_next = gate_next * align_next * sfactor_next

    settle = 3.0 * (phi_next - phi_prev)

    # ---------- assemble ----------
    components = {
        "approach_cargo": float(approach_cargo),
        "progress": float(progress),
        "dock_enter": float(dock_enter),
        "settle": float(settle),
        "roughness": float(roughness),
        "action_cost": float(action_cost),
        "time_cost": float(time_cost),
        "hard_hit": float(hard_hit),
        "terminal_success": float(terminal_success),
        "terminal_failure": float(terminal_failure),
    }

    total_reward = (
        approach_cargo
        + progress
        + dock_enter
        + settle
        + roughness
        + action_cost
        + time_cost
        + hard_hit
        + terminal_success
        + terminal_failure
    )
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=309.545139, len=180.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[309.120308, 310.570189]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| terminal_success | 300.000000 | 95.5% | 95.5% | 0.6% |
| dock_enter | 5.000000 | 1.6% | 1.6% | 0.6% |
| progress | 4.070160 | 1.3% | 1.3% | 78.8% |
| settle | 2.593036 | 0.8% | 0.8% | 20.4% |
| approach_cargo | 1.094736 | 0.3% | 0.6% | 99.3% |
| time_cost | -0.361800 | -0.1% | 0.1% | 100.0% |
| action_cost | -0.077889 | -0.0% | 0.0% | 100.0% |
| roughness | -0.044508 | -0.0% | 0.0% | 20.7% |
| hard_hit | 0.000000 | 0.0% | 0.0% | 0.0% |
| terminal_failure | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个俯视视角的仓库非预hensile操作任务：一辆只能施加纵向驱动力和转向力矩、没有刹车、没有夹爪的轮式小车，需要把地面上的一个可自由滑动且易碎的方形货箱，通过推撞方式从隔墙近侧运送到远侧的交付坞。主目标是让货箱完全进入坞内、货箱朝向与坞对齐、货箱接近静止，并持续一小段稳定时间；次目标包括避免货箱受到重复硬冲击、避免小车或货箱离开仓库地面、避免撞墙卡死。不该混淆的目标：小车自身到达坞、货箱仅仅碰到坞区域、货箱高速冲过坞、用持续推挤阻止货箱滑动，这些都不等于任务成功。小车无法从后方给货箱刹车，货箱只能靠地面阻尼减速。

## 3. 观察空间 observation_space
- type: Box
- shape: [19]
- dtype: float32
- bounds: 全部裁剪到 [-2.0, 2.0]；部分维度是归一化量，极值可能饱和。
- obs[0]: cart_x，小车 x 位置 / 仓库半宽（5.0），0 为中线，+1 为远墙侧；reward_usable: true
- obs[1]: cart_y，小车 y 位置 / 仓库半高（4.0），0 为中线；reward_usable: true
- obs[2]: cart_cos_heading，小车朝向余弦；reward_usable: true
- obs[3]: cart_sin_heading，小车朝向正弦；reward_usable: true
- obs[4]: cart_forward_speed，小车沿自身朝向速度 / 3.0 m/s；reward_usable: true
- obs[5]: cart_yaw_rate，小车角速度 / 8.0 rad/s；reward_usable: true
- obs[6]: crate_rel_x_body，货箱相对小车在车体坐标系下的 x 分量 / 3.0 m；reward_usable: true
- obs[7]: crate_rel_y_body，货箱相对小车在车体坐标系下的 y 分量 / 3.0 m；reward_usable: true
- obs[8]: crate_vx，货箱世界系 x 速度 / 3.0 m/s；reward_usable: true
- obs[9]: crate_vy，货箱世界系 y 速度 / 3.0 m/s；reward_usable: true
- obs[10]: crate_cos_heading，货箱朝向余弦；reward_usable: true
- obs[11]: crate_sin_heading，货箱朝向正弦；reward_usable: true
- obs[12]: crate_to_dock_x，货箱中心到坞中心的有符号 x 偏移 / 仓库半宽（5.0）；reward_usable: true
- obs[13]: crate_to_dock_y，货箱中心到坞中心的有符号 y 偏移 / 仓库半高（4.0）；reward_usable: true
- obs[14]: cart_crate_contact，小车与货箱当前是否接触，1.0 为接触，0.0 为不接触；reward_usable: true
- obs[15]: sensor_front，小车前方最近静态障碍接近度，0 为范围内清晰，1 为接触；reward_usable: true
- obs[16]: sensor_left，小车左侧最近静态障碍接近度，0 为范围内清晰，1 为接触；reward_usable: true
- obs[17]: sensor_right，小车右侧最近静态障碍接近度，0 为范围内清晰，1 为接触；reward_usable: true
- obs[18]: time_fraction，已消耗 episode 时间预算比例，范围 [0,1]；reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: [2]
- continuous: true
- bounds: [-1.0, 1.0] per channel
- action 0: drive，沿小车朝向的纵向力命令；+1 前进，-1 倒退。
- action 1: steer，转向力矩命令；+1 左转/逆时针，-1 右转。
- 无刹车动作，无夹爪动作。小车无法直接制动货箱，货箱只靠地面阻尼减速。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
  - docked_success：货箱完全进入坞内，货箱朝向误差小于 30 度，货箱速度小于 0.05 m/s，并且这些条件连续保持 10 个环境步。
  - “完全进入坞内”的精确几何：坞为 0.84 m × 0.84 m，货箱为 0.60 m × 0.60 m，因此要求 |crate_to_dock_x| <= 0.12 m 且 |crate_to_dock_y| <= 0.12 m，即 |obs[12]| <= 0.024 且 |obs[13]| <= 0.030。
  - 成功终止立即结束，之后没有额外时间累积奖励。
- failure-like termination:
  - crate_out_of_bounds：货箱中心离开仓库地面矩形。
  - cart_out_of_bounds：小车中心离开仓库地面矩形。
  - crate_damaged：货箱承受 3 次或更多硬冲击；硬冲击定义为小车-货箱接触的峰值法向冲量超过易碎阈值。
- ambiguous termination:
  - 仅有 episode 结束但无法从 info 读取原因时，需要通过 obs 间接推断，不是显式可读信号。
- truncation:
  - time_limit：达到固定步数预算；报告为截断，不是任务成功。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []
- forbidden_or_uncertain_info_fields:
  - is_success
  - cargo_goal_distance
  - cargo_angle_error
  - cargo_speed
  - robot_cargo_distance
  - contact_impulse
  - hard_collision_count
  - stagnation_steps
  - action_energy
  - component_returns
  - official_reward_terms
  - termination_reason
  - cargo_inside_dock
  - stable_steps

## 7. 可用于奖励函数的信号
- position:
  - obs[0], obs[1]：小车归一化位置，可用于边界约束。
  - obs[12], obs[13]：货箱到坞中心的归一化偏移，可直接推导货箱到坞距离和是否完全进入坞。
  - obs[6], obs[7] + obs[0], obs[1], obs[2], obs[3]：可恢复货箱世界坐标；用于货箱出界推断、相对位置分析。
  - derived_possible：货箱是否完全进入坞内，可由 |obs[12]| <= 0.024 且 |obs[13]| <= 0.030 推断。
  - derived_possible：货箱或小车是否接近/越过仓库边界，可由 obs[0], obs[1] 以及恢复出的货箱世界坐标推断。
- velocity:
  - obs[4]：小车纵向速度，可用于接触冲击代理、推动阶段控制。
  - obs[5]：小车角速度，可用于转向稳定性分析。
  - obs[8], obs[9]：货箱世界系速度，可计算货箱速度大小 sqrt((obs[8]*3.0)^2 + (obs[9]*3.0)^2)。
  - derived_possible：货箱低速稳定条件，可用货箱速度 < 0.05 m/s 推断。
  - derived_possible：小车-货箱相对速度/碰撞烈度代理，可用 obs[14] 接触标记结合 obs[4], obs[8], obs[9] 推断，但无法恢复真实峰值法向冲量。
- orientation:
  - obs[2], obs[3]：小车朝向，用于坐标变换。
  - obs[10], obs[11]：货箱朝向；接口说明中货箱朝向误差代理为 atan2(obs[11], obs[10])。
  - derived_possible：货箱朝向误差是否小于 30 度，可由 atan2(obs[11], obs[10]) 推断。
- contact:
  - obs[14]：小车-货箱是否接触。
  - obs[15], obs[16], obs[17]：静态障碍接近度，可用于隔墙/边界避碰。
  - derived_possible：硬冲击事件代理，可由接触发生时刻的相对速度/货箱速度突变推断，但阈值和计数不可精确读取。
- action/engine:
  - action[0]：纵向力命令，可用于平滑/推力控制，但当前任务没有明确能耗或动作幅度要求。
  - action[1]：转向命令，可用于转向平滑控制。
- other:
  - obs[18]：时间预算消耗比例，可用于时间效率或截断风险分析。
  - derived_possible：成功终止可间接推断为货箱完全进入坞、朝向对齐、速度极低并持续若干步；但持续步数需要内部计数器，不能从单帧 obs 直接读取。

# 6.5. 已知的奖励结构（本环境的作者奖励分项语义与权重；优先于上文任何与之冲突的通用规则）
<!-- Generated by tools/extract_v9_structure_block.py — do not edit by hand.

     Provenance: the block below is extracted VERBATIM from
     `prompts/eureka_01_initial_reward_v9.md`, the prompt whose measured effect was
     4/60 -> 58/60 (runs/env_007/V9_STRUCTURE_FINDINGS.md). The CREATE pipeline injects it
     into the environment card so that CREATE and EUREKA receive the same reward
     information; see runs/env_007/CREATE_V9_SPEC_AND_STRUCTURE.md for the protocol and the
     declared asymmetry against the older CREATE baseline.

     Source sha256: bb3e15cd11343ee30f07aca73c35ca289d11ce9aa4da8b006ff272e9de8c16be
     Block sha256 : f3fca2aaeb9157bc8cd163fd605e87c921170f76ab073de928ea387eb99e0bac
-->

<!-- BEGIN v9 structure block (verbatim) -->

# 已知的奖励结构（本版本新增，**必须照此实现**）

本环境的作者奖励函数是已知的，其**分项语义与权重**如下。你的任务是：**用 `obs` 与 `action` 把它重写出来。**
（注意：这些是"设计结构"，不是"可以直接读取的字段"——你**不能**读 `info`，也拿不到任何未在观测中声明的量。）

| 分项 | 权重 | 语义 |
|---|---:|---|
| `approach_cargo` | **+1.0 / 米** | 本帧小车→货箱距离缩短了多少米（势函数形式） |
| `progress` | **+1.0 / 米** | 本帧货箱→泊位距离缩短了多少米（势函数形式） |
| `dock_enter` | **+5.0** | 货箱**首次**完全进入泊位容差时，一次性发放 |
| `roughness` | **−0.02 / (N·s)** | 接触冲量比例惩罚：撞得越狠罚越多（"轻拿轻放"） |
| `action_cost` | **−0.0005** | 动作平方和 |
| `time_cost` | **−0.002 / 步** | 每步固定时间成本 |
| `hard_hit` | **−0.5** | 单步内发生一次硬冲击的固定惩罚 |
| `terminal_success` | **+300** | 完成条件**连续 10 步**成立时发放一次 |
| `terminal_failure` | **−100** | 货箱或小车越界、或累计硬冲击 ≥3 次时发放一次 |

把这些分项重写成 `obs` + `action` 的表达式时，逐项的可行性与做法如下：

1. **`approach_cargo`**：小车→货箱距离（由 `obs[6]`, `obs[7]` 与姿态还原）**本帧的差**，按米计。
2. **`progress`**：货箱→泊位距离（由 `obs[12]`, `obs[13]` 还原）**本帧的差**，按米计。
   **两项都必须是有符号、对称的**：靠近给正分，远离给负分。
3. **`dock_enter`**：用模块级状态记录"是否曾经完全进入容差"，只发一次。
4. **`action_cost` / `time_cost`**：直接照写。
5. **`terminal_success`**：用模块级状态对"泊位内 + 对齐 + 慢"做**连续计数**，达到 10 次时发一次。
   注意本环境在满足时会**立即终止** episode，所以它天然最多只发一次。
6. **`terminal_failure`**：越界由 `obs[0]`/`obs[1]` 判断（`|x|` 或 `|y|` 超过约 1.05 即出界）。
7. **`roughness`（唯一需要代理的项）**：观测里**没有冲量**这一维。请构造一个可观测的
   "接近速度 × 接触"代理——例如
   `closing = obs[4]*3.0 - (obs[8]*3.0*obs[2] + obs[9]*3.0*obs[3])`，再取 `max(0, closing)`，
   仅在实际接触（`obs[14] > 0.5`）时乘以一个负系数。**注意两点**：
   该项是用来**抑制撞击**的，不是用来压制推进的——实测把它的量级做到与推进项相当，
   会让策略连动都不敢动（入坞率 0.00）。**它必须比推进项弱**，只在"撞得很快"时显著。

**优先级声明**：本节给出的分项表**优先于**下文任何与之冲突的通用规则；若下文的规则与本节冲突，以本节为准。

<!-- END v9 structure block -->