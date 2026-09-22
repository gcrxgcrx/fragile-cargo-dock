def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v1 for 2D Lander goal reaching with soft landing.
    
    Design hypothesis: The previous attempts used proximity + contact bonus + landing penalty
    and stagnated at ~5.7. The hypothesis is that contact bonus alone was sparse and possibly
    exploited, while the proximity reward lacked velocity-conditional gating and orientation
    coupling. This version shifts to:
    - delta_distance as the primary progress driver (dense, per-step gradient)
    - bounded absolute distance as a shaping complement (prevents drifting)
    - velocity+angular constraints via hinge penalties (only when dangerous)
    - orientation+angular velocity quadratic suppression (continuous guidance)
    - minimal fuel penalty (no conditional gating, just light discouragement)
    
    This materially differs from the tried structure by:
    1) Using improvement_delta as the main progress signal instead of absolute proximity
    2) Removing discrete contact bonus entirely (replaced by implicit soft-landing shaping)
    3) Using hinge penalties for velocity instead of quadratic (gives freedom in safe range)
    """
    
    # --- Unpack observations ---
    # Current state
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    
    # Next state
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_body_angle = next_obs[4]
    n_angular_vel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]
    
    # --- Component A: delta_distance (primary progress driver) ---
    # Encourage reducing distance to target pad center (0, 0)
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (nx_pos**2 + ny_pos**2) ** 0.5
    delta_distance = current_distance - next_distance
    
    # Scale: small positive reward for approaching, near-zero when stationary
    progress_reward = 10.0 * delta_distance
    
    # --- Component B: bounded_distance_shaping (absolute proximity guidance) ---
    # Complement to delta: ensures agent doesn't drift far even if delta is zero
    # Use 1/(1+k*d) to give strong gradient near target, saturates at large distances
    distance_shaping = 2.0 * (1.0 / (1.0 + 0.5 * next_distance))
    
    # --- Component C: velocity_hinge_constraint (soft safety on speed) ---
    # Penalize horizontal speed when too high (over 2.0 m/s)
    h_speed = abs(nx_vel)
    h_penalty = max(0.0, h_speed - 2.0)
    
    # Penalize vertical speed: downward too fast (> -1.5, penalize magnitude),
    # upward too fast (> 1.0, penalize)
    v_speed = ny_vel
    v_down_penalty = max(0.0, -v_speed - 1.5)  # penalize when more negative than -1.5
    v_up_penalty = max(0.0, v_speed - 1.0)     # penalize when more positive than 1.0
    
    # Angular velocity penalty: penalize when |angular_vel| > 0.8
    angular_penalty = max(0.0, abs(n_angular_vel) - 0.8)
    
    velocity_constraint = -0.5 * (h_penalty + v_down_penalty + v_up_penalty + angular_penalty)
    
    # --- Component D: orientation_stabilization (quadratic on angle + angular vel) ---
    # Continuous penalty for tilt and rotation, light enough to not freeze the agent
    orientation_penalty = -0.3 * (n_body_angle**2) - 0.2 * (n_angular_vel**2)
    
    # --- Component E: fuel_efficiency (light action penalty) ---
    # Encourage no_engine (action 0) when possible
    # action 0: no_engine, 1: left_orientation, 2: main_engine, 3: right_orientation
    if action == 0:
        fuel_penalty = 0.0
    elif action == 2:
        fuel_penalty = -0.08  # main engine: slightly more expensive
    else:
        fuel_penalty = -0.05  # orientation engines: light penalty
    
    # --- Assemble total reward ---
    total_reward = (
        progress_reward +
        distance_shaping +
        velocity_constraint +
        orientation_penalty +
        fuel_penalty
    )
    
    components = {
        'progress_reward': progress_reward,
        'distance_shaping': distance_shaping,
        'velocity_constraint': velocity_constraint,
        'orientation_penalty': orientation_penalty,
        'fuel_penalty': fuel_penalty
    }
    
    return float(total_reward), components