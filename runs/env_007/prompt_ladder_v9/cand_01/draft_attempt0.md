```python
# =============================================================================
# Module-level state (mutable containers instead of `global`)
# =============================================================================
_PREV_T = [-1.0]       # previous obs[18] value -> episode-boundary detection
_STREAK = [0]          # consecutive steps satisfying the docking predicate
_PAID = [False]        # one-shot terminal success already paid this episode
_ENTERED = [False]     # one-shot "first fully inside the dock" already paid
_FAIL_PAID = [False]   # one-shot terminal failure already paid this episode
_HARD_HITS = [0]       # de-bounced hard-impact count this episode
_PREV_HARD = [False]   # previous-step hard-impact flag (rising-edge detection)

# -----------------------------------------------------------------------------
# Geometry / thresholds taken from the environment facts (no invented values)
# -----------------------------------------------------------------------------
HALF_W = 5.0        # warehouse half width  [m] (obs[0], obs[12] normalised by it)
HALF_H = 4.0        # warehouse half height [m] (obs[1], obs[13] normalised by it)
REL_M = 3.0         # scale of obs[6], obs[7] (cart-frame crate offset, metres)
DOCK_TOL_X = 0.024  # |obs[12]| <= 0.024  -> crate completely inside the dock (x)
DOCK_TOL_Y = 0.030  # |obs[13]| <= 0.030  -> crate completely inside the dock (y)
ALIGN_COS = 0.8660254  # cos(30 deg): crate heading error < 30 deg
SLOW_V = 0.05       # crate speed threshold for "practically at rest" [m/s]
HARD_CLOSING = 1.0  # contact closing speed [m/s] above which an impact counts hard
BOUND_LO = 0.95     # cart guard starts to bite here (|obs[0]| or |obs[1]|)
BOUND_HI = 1.0      # crate guard starts to bite here (normalised crate coord)
OOB_LEVEL = 1.05    # declared out-of-bounds level

# Weights of the known reward structure (kept verbatim)
W_APPROACH = 1.0            # +1.0 / m  (cart -> crate distance shrink)
W_PROGRESS = 1.0            # +1.0 / m  (crate -> dock distance shrink)
W_DOCK_ENTER = 5.0          # +5.0 once, first time fully inside the tolerance
ROUGH_K = 0.05              # roughness proxy: -k * contact * closing
W_ACTION_COST = 0.0005      # -0.0005 * sum(action^2)
W_TIME_COST = 0.002         # -0.002 per step
W_HARD_HIT = -0.5           # -0.5 per step containing one hard impact
W_SUCCESS_EVENT = 300.0     # +300 once, after 10 consecutive docking steps
W_FAIL_EVENT = -100.0       # -100 once, on out-of-bounds / >= 3 hard impacts
W_SETTLED_STEP = 1.0        # per-step income while the docking predicate holds


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------------------------------------------------------------
    # Numeric self-checks (dt ~ 0.05 s, steady pushing speed ~1 m/s -> push term
    # ~0.05 per step):
    #   1) idle    : approach 0, progress 0, action 0        -> ~ -0.002
    #      pushing : approach ~0 + progress ~ +0.05          -> ~ +0.048  > idle  OK
    #   2) hovering 0.3 m outside the dock for 400 steps      -> ~ -0.8
    #      really docking + 10 hold steps: progress (+0.3) + dock_enter (+5)
    #      + 10 * (+1.0 settled) + success event (clipped to +20)  >> -0.8       OK
    #   3) contact, closing = 1.00 m/s : roughness -0.050, hard_hit -0.500
    #      contact, closing = 0.05 m/s : roughness -0.0025, hard_hit  0.000
    #      difference ~0.55 >> push term ~0.05                                   OK
    #   4) same settled state called 12x: every call pays +1.0 (settled_hold),
    #      it is never switched off by the one-shot events                       OK
    #   5) R_idle ~ -0.002  <  R_push ~ +0.048  <  R_settled ~ +1.00             OK
    #      (in the steady-push state closing ~ 0, so no roughness/hard_hit is
    #       charged: the effective per-step penalty is only ~0.003)
    #   6) bounds: cart at centre (0,0) -> 0.0 ; cart at |obs[0]| = 1.05 -> -2.6 OK
    # -------------------------------------------------------------------------
    t = float(next_obs[18])
    if t < _PREV_T[0] or t <= 1.0 / 400.0:      # new episode detected
        _STREAK[0] = 0
        _PAID[0] = False
        _ENTERED[0] = False
        _FAIL_PAID[0] = False
        _HARD_HITS[0] = 0
        _PREV_HARD[0] = False
    _PREV_T[0] = t

    # -------------------------------------------------------------------------
    # 1) approach_cargo: cart -> crate distance shrink this frame, signed, metres
    #    (the body-frame offset norm is frame invariant)
    # -------------------------------------------------------------------------
    d_cart_crate_prev = REL_M * ((obs[6] * obs[6] + obs[7] * obs[7]) ** 0.5)
    d_cart_crate_next = REL_M * ((next_obs[6] * next_obs[6] + next_obs[7] * next_obs[7]) ** 0.5)
    approach_cargo = W_APPROACH * (d_cart_crate_prev - d_cart_crate_next)

    # -------------------------------------------------------------------------
    # 2) progress: crate -> dock distance shrink this frame, signed, metres
    # -------------------------------------------------------------------------
    dx_prev = obs[12] * HALF_W
    dy_prev = obs[13] * HALF_H
    dx_next = next_obs[12] * HALF_W
    dy_next = next_obs[13] * HALF_H
    d_dock_prev = (dx_prev * dx_prev + dy_prev * dy_prev) ** 0.5
    d_dock_next = (dx_next * dx_next + dy_next * dy_next) ** 0.5
    progress = W_PROGRESS * (d_dock_prev - d_dock_next)

    # -------------------------------------------------------------------------
    # 3) dock_enter: crate completely inside the dock tolerance, paid once
    # -------------------------------------------------------------------------
    inside_now = (abs(next_obs[12]) <= DOCK_TOL_X) and (abs(next_obs[13]) <= DOCK_TOL_Y)
    dock_enter = 0.0
    if inside_now and not _ENTERED[0]:
        _ENTERED[0] = True
        dock_enter = W_DOCK_ENTER

    # -------------------------------------------------------------------------
    # 4) roughness: observable proxy of the contact impulse
    #    (closing speed along the cart heading, only while touching)
    # -------------------------------------------------------------------------
    crate_vx = next_obs[8] * 3.0
    crate_vy = next_obs[9] * 3.0
    crate_along_heading = crate_vx * obs[2] + crate_vy * obs[3]
    closing = obs[4] * 3.0 - crate_along_heading
    if closing < 0.0:
        closing = 0.0
    contact = 1.0 if next_obs[14] > 0.5 else 0.0
    roughness = -ROUGH_K * contact * closing

    # hard impact proxy: touching AND still closing fast
    hard_now = (contact > 0.5) and (closing >= HARD_CLOSING)
    hard_hit = W_HARD_HIT if hard_now else 0.0
    if hard_now and not _PREV_HARD[0]:       # rising edge -> count one impact
        _HARD_HITS[0] += 1
    _PREV_HARD[0] = hard_now

    # -------------------------------------------------------------------------
    # 5) action cost / time cost
    # -------------------------------------------------------------------------
    action_cost = -W_ACTION_COST * (action[0] * action[0] + action[1] * action[1])
    time_cost = -W_TIME_COST

    # -------------------------------------------------------------------------
    # 6) bounds guard: cart position (obs[0], obs[1]) and reconstructed crate
    #    world position; monotone decrease as the wall is approached
    # -------------------------------------------------------------------------
    cos_h = next_obs[2]
    sin_h = next_obs[3]
    rel_x = next_obs[6] * REL_M
    rel_y = next_obs[7] * REL_M
    crate_wx = next_obs[0] * HALF_W + rel_x * cos_h - rel_y * sin_h
    crate_wy = next_obs[1] * HALF_H + rel_x * sin_h + rel_y * cos_h
    crate_nx = crate_wx / HALF_W
    crate_ny = crate_wy / HALF_H

    bounds_guard = 0.0
    ex = abs(next_obs[0]) - BOUND_LO
    if ex > 0.0:
        bounds_guard += -20.0 * ex - 60.0 * ex * ex
    ey = abs(next_obs[1]) - BOUND_LO
    if ey > 0.0:
        bounds_guard += -20.0 * ey - 60.0 * ey * ey
    ecx = abs(crate_nx) - BOUND_HI
    if ecx > 0.0:
        bounds_guard += -20.0 * ecx - 60.0 * ecx * ecx
    ecy = abs(crate_ny) - BOUND_HI
    if ecy > 0.0:
        bounds_guard += -20.0 * ecy - 60.0 * ecy * ecy

    # -------------------------------------------------------------------------
    # 7) docking predicate -> per-step settled income + one-shot success event
    #    predicate: fully inside the dock + heading aligned (<30 deg) + at rest
    # -------------------------------------------------------------------------
    crate_speed = ((next_obs[8] * 3.0) ** 2 + (next_obs[9] * 3.0) ** 2) ** 0.5
    aligned = next_obs[10] >= ALIGN_COS
    slow = crate_speed < SLOW_V
    settled_now = inside_now and aligned and slow

    if settled_now:
        _STREAK[0] += 1
    else:
        _STREAK[0] = 0

    # paid on EVERY step the predicate holds (never gated by the one-shot events)
    settled_hold = W_SETTLED_STEP if settled_now else 0.0

    success_event = 0.0
    if _STREAK[0] >= 10 and not _PAID[0]:    # one-shot, per episode
        _PAID[0] = True
        success_event = W_SUCCESS_EVENT

    # -------------------------------------------------------------------------
    # 8) terminal failure: cart / crate out of bounds, or >= 3 hard impacts
    # -------------------------------------------------------------------------
    out_of_bounds = (
        abs(next_obs[0]) > OOB_LEVEL
        or abs(next_obs[1]) > OOB_LEVEL
        or abs(crate_nx) > OOB_LEVEL
        or abs(crate_ny) > OOB_LEVEL
        or _HARD_HITS[0] >= 3
    )
    fail_event = 0.0
    if out_of_bounds and not _FAIL_PAID[0]:
        _FAIL_PAID[0] = True
        fail_event = W_FAIL_EVENT

    # -------------------------------------------------------------------------
    # total
    # -------------------------------------------------------------------------
    total = (
        approach_cargo
        + progress
        + dock_enter
        + roughness
        + action_cost
        + time_cost
        + hard_hit
        + settled_hold
        + success_event
        + fail_event
        + bounds_guard
    )

    components = {
        "approach_cargo": approach_cargo,
        "progress": progress,
        "dock_enter": dock_enter,
        "roughness": roughness,
        "action_cost": action_cost,
        "time_cost": time_cost,
        "hard_hit": hard_hit,
        "settled_hold": settled_hold,
        "terminal_success": success_event,
        "terminal_failure": fail_event,
        "bounds_guard": bounds_guard,
    }
    return float(total), components
```