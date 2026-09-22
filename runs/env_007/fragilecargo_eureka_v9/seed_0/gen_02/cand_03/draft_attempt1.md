**分析**：当前奖励的 93% 回报来自 `settle_bonus`（+2.0/步的泊位悬停）与 `terminal_success`，而真正的运输进展（`progress`）只占约 1.4%，但 20/20 评测回合全部 truncate、0 次真实终止 —— 说明代理判据（中心距 <0.25 m + |cos|≥0.866 + 慢速）系统性地比环境判据宽松，智能体在"伪成功"状态里长期刷分，训练回报饱和（≈410）而原生任务分停在 3.9。改进方向：① 把平坦悬停奖励换成**严格位置（逐轴 ≤0.15 m）**驱动的连续联合因子，去掉"半对齐赖着收分"的悬崖；② 一次性成功事件改用**带符号朝向**（crate 朝 +x）这一更贴近"朝向误差"定义的门控；③ `roughness` 改成只在真正高速闭合时生效的 hinge，保证稳态推箱不被罚；④ 降低越界/动作/时间成本权重，避免压过主信号。

```python
# self-check (ASCII, single-step estimates)
#   geometry: warehouse half width 5.0 m, half height 4.0 m; obs[6..9] / 3.0
#   IDLE  (do nothing, crate parked away from dock):
#         progress 0, approach 0, settle 0, roughness 0, time -0.001, action ~0
#         -> total ~ -0.001
#   PUSH  (crate is being driven toward the dock, so the crate HAS velocity;
#          cart in steady contact, closing speed ~ 0 < hinge 0.6):
#         progress +2.5*0.03 = +0.075, approach ~0, roughness 0, costs ~ -0.0015
#         -> total ~ +0.073  STRICTLY GREATER than IDLE
#   No auxiliary term can go negative while pushing: settle is a product of
#   non-negative factors (only adds), roughness/hard_hit require closing > 0.6
#   (a slam, never the steady push), bounds_safety only near the walls.
#   DOCKED (crate centered, speed 0, axis aligned): settle ~ +1.2/step, and
#   with signed heading cos >= 0.866 held 10 steps -> one-shot +250.

_PREV_T = [-1.0]
_STREAK = [0]
_PAID_SUCCESS = [False]
_PAID_ENTER = [False]
_HARD_HITS = [0]
_PAID_FAIL = [False]


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 0.0025:
        _STREAK[0] = 0
        _PAID_SUCCESS[0] = False
        _PAID_ENTER[0] = False
        _HARD_HITS[0] = 0
        _PAID_FAIL[0] = False
    _PREV_T[0] = t

    cart_x = obs[0] * 5.0
    cart_y = obs[1] * 4.0
    c = obs[2]
    s = obs[3]

    rel_x = obs[6] * 3.0
    rel_y = obs[7] * 3.0
    d_cc = (rel_x * rel_x + rel_y * rel_y) ** 0.5
    nrel_x = next_obs[6] * 3.0
    nrel_y = next_obs[7] * 3.0
    nd_cc = (nrel_x * nrel_x + nrel_y * nrel_y) ** 0.5

    ddx = obs[12] * 5.0
    ddy = obs[13] * 4.0
    d_dock = (ddx * ddx + ddy * ddy) ** 0.5
    nddx = next_obs[12] * 5.0
    nddy = next_obs[13] * 4.0
    nd_dock = (nddx * nddx + nddy * nddy) ** 0.5

    crate_x = cart_x + c * rel_x - s * rel_y
    crate_y = cart_y + s * rel_x + c * rel_y

    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_speed = (crate_vx * crate_vx + crate_vy * crate_vy) ** 0.5
    cos_crate = next_obs[10]

    # 1) main transport signal: net reduction of crate -> dock distance
    progress = 2.5 * (d_dock - nd_dock)

    # 2) early shaping: cart closing on the crate (signed delta, not farmable)
    approach_cargo = 0.5 * (d_cc - nd_cc)

    # 3) one-shot milestone: crate first reaches the dock neighbourhood
    dock_enter = 0.0
    if nd_dock < 0.35 and not _PAID_ENTER[0]:
        _PAID_ENTER[0] = True
        dock_enter = 2.0

    # 4) joint docking quality (near * slow * axis aligned), non-negative only
    near = 1.0 - nd_dock / 0.6
    if near < 0.0:
        near = 0.0
    if near > 1.0:
        near = 1.0
    slow = 1.0 - crate_speed / 0.6
    if slow < 0.0:
        slow = 0.0
    if slow > 1.0:
        slow = 1.0
    acos = cos_crate
    if acos < 0.0:
        acos = -acos
    align = (acos - 0.5) / 0.45
    if align < 0.0:
        align = 0.0
    if align > 1.0:
        align = 1.0
    settle_bonus = 1.2 * ((near * slow * align) ** 0.3333333)

    # 5) contact roughness: only for high closing speed (a slam), zero in a
    #    steady push where crate and cart move together
    along = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - along
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0

    roughness = 0.0
    if contact > 0.5 and closing > 0.6:
        roughness = -0.15 * (closing - 0.6)

    hard_hit = 0.0
    if contact > 0.5 and closing > 1.5:
        hard_hit = -0.25
        _HARD_HITS[0] += 1

    action_cost = -0.0004 * (action[0] * action[0] + action[1] * action[1])
    time_cost = -0.001

    # 6) bounds guard (hinge, only close to / beyond the floor)
    cart_margin = abs(obs[0])
    if abs(obs[1]) > cart_margin:
        cart_margin = abs(obs[1])
    crate_margin = abs(crate_x) / 5.0
    if abs(crate_y) / 4.0 > crate_margin:
        crate_margin = abs(crate_y) / 4.0
    m = cart_margin
    if crate_margin > m:
        m = crate_margin
    bounds_safety = 0.0
    if m > 0.92:
        over = m - 0.92
        bounds_safety = -(2.0 * over + 50.0 * over * over)

    # 7) strict success proxy: tight per-axis centring, settled, heading facing
    #    the far side (+x, the direction of transport) held for 10 steps
    pos_ok = 0.0
    if abs(nddx) <= 0.15 and abs(nddy) <= 0.15:
        pos_ok = 1.0
    slow_ok = 0.0
    if crate_speed < 0.05:
        slow_ok = 1.0
    sign_ok = 0.0
    if cos_crate >= 0.866:
        sign_ok = 1.0

    if pos_ok > 0.5 and slow_ok > 0.5 and sign_ok > 0.5:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    terminal_success = 0.0
    if _STREAK[0] >= 10 and not _PAID_SUCCESS[0]:
        _PAID_SUCCESS[0] = True
        terminal_success = 250.0

    terminal_failure = 0.0
    if not _PAID_FAIL[0]:
        if cart_margin > 1.05 or crate_margin > 1.0 or _HARD_HITS[0] >= 3:
            _PAID_FAIL[0] = True
            terminal_failure = -50.0

    components = {
        "progress": progress,
        "approach_cargo": approach_cargo,
        "dock_enter": dock_enter,
        "settle_bonus": settle_bonus,
        "roughness": roughness,
        "hard_hit": hard_hit,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "bounds_safety": bounds_safety,
        "terminal_success": terminal_success,
        "terminal_failure": terminal_failure,
    }

    total = (progress + approach_cargo + dock_enter + settle_bonus + roughness
             + hard_hit + action_cost + time_cost + bounds_safety
             + terminal_success + terminal_failure)

    return (float(total), components)
```