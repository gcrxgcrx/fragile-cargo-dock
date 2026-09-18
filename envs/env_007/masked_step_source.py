def step(self, action):
    # Top-down rigid-body warehouse simulation (Box2D), gravity is zero and the
    # floor is emulated with linear/angular damping.
    #
    # Scene: one cart, one freely-moving fragile crate, an interior partition
    # wall with a single narrow opening, and a marked docking bay on the far
    # side of the wall.
    #
    # Per step the cart receives a longitudinal force along its heading and a
    # steering torque:
    #     throttle, steer = action[0], action[1]
    #     force  = heading_vector * throttle * MAX_FORCE
    #     torque = steer * MAX_TORQUE
    #
    # The crate can only be moved by contact with the cart; there is no gripper.
    # The cart has no brake, so the crate keeps gliding after the cart releases
    # it and only slows because of floor drag.
    #
    # The simulation is advanced for FRAME_SKIP physics sub-steps per
    # environment step. A contact listener records the peak normal impulse of
    # cart<->crate contacts during the step.

    for _ in range(FRAME_SKIP):
        cart.ApplyForceToCenter(force)
        cart.ApplyTorque(torque)
        world.Step(PHYSICS_DT, 8, 3)

    # Contact and fragility bookkeeping.
    contact = cart_and_crate_are_touching
    peak_impulse = max_normal_impulse_of_cart_crate_contacts_this_step
    if peak_impulse > FRAGILITY_IMPULSE_THRESHOLD:
        hard_collision_count += 1

    # Task progress bookkeeping.
    crate_to_dock_distance = distance(crate_position, dock_centre)
    progress = previous_crate_to_dock_distance - crate_to_dock_distance
    stagnation_steps = stagnation_steps + 1 if progress < STAGNATION_EPS else 0
    crate_inside_dock = crate_is_fully_contained_in_dock_rectangle
    newly_docked = crate_inside_dock and not dock_was_entered_before
    stable_now = (
        crate_inside_dock
        and crate_heading_error < 0.5236          # 30 degrees
        and crate_linear_speed < 0.05             # m/s
    )
    stable_steps = stable_steps + 1 if stable_now else 0

    # Termination.
    terminated = (
        stable_steps >= 10                        # docking held long enough
        or crate_centre_outside_warehouse
        or cart_centre_outside_warehouse
        or hard_collision_count >= 3
    )
    truncated = elapsed_steps >= MAX_EPISODE_STEPS

    masked_reward = <OFFICIAL_REWARD_MASKED>

    # The simulation above exposes physical quantities in info (crate-to-dock
    # distance, heading error, crate speed, contact impulse, hard-collision
    # count, stagnation counter, action energy, per-term reward returns and the
    # termination reason). Those fields are for logging and for the official
    # evaluation only; they are masked and forbidden for generated reward
    # functions in this experiment.
    info = {
        "is_success": ..., "cargo_goal_distance": ..., "cargo_angle_error": ...,
        "cargo_speed": ..., "robot_cargo_distance": ..., "contact_impulse": ...,
        "hard_collision_count": ..., "stagnation_steps": ..., "action_energy": ...,
        "cargo_inside_dock": ..., "stable_steps": ..., "termination_reason": ...,
    }
    return observation, masked_reward, terminated, truncated, info
