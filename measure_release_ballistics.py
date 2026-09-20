"""How much 'lead' does the release need, and how sensitive is it?

The cart has no brake, so the delivery is ballistic: the cart must release the
crate at some point and let floor drag stop it inside the dock. This measures
the two quantities that decide how hard that is, by giving the crate a known
initial speed on the far side of the partition (no wall in the way), moving the
cart out of contact, and letting the crate coast until it is slower than the
0.05 m/s success threshold.

Reported per initial speed:
  * coast distance  - how far ahead of the stopping point the release must be
  * steps to stop   - how long the settle takes
  * timing margin   - the dock tolerance (+/-0.12 m) expressed in control steps
                      at that speed, i.e. how precisely the release must be timed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import custom_envs.registration  # noqa: F401
import gymnasium as gym
import numpy as np

ENV_ID = "FragileCargoDock-v0"
DOCK_TOL = 0.12          # |crate - dock centre| must be <= this to count as inside
SPEED_TOL = 0.05         # success threshold
DT = 1.0 / 30.0
START_X = 1.10           # far side of the partition, on the approach run to the dock


def main() -> None:
    env = gym.make(ENV_ID)
    env.reset(seed=0)
    base = env.unwrapped
    print(f"{'v0 (m/s)':>9} {'coast (m)':>10} {'steps':>7} {'release at x':>13} "
          f"{'window (m)':>11} {'margin (steps)':>15}")
    for v0 in (0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0):
        env.reset(seed=0)
        # park the cart in a far corner and keep it there
        base._robot_body.position = (-4.0, 3.0)
        base._robot_body.linearVelocity = (0.0, 0.0)
        base._robot_body.angularVelocity = 0.0
        # launch the crate
        base._cargo_body.position = (START_X, 0.0)
        base._cargo_body.angle = 0.0
        base._cargo_body.angularVelocity = 0.0
        base._cargo_body.linearVelocity = (v0, 0.0)
        steps = 0
        while steps < 1200:
            base._robot_body.position = (-4.0, 3.0)
            base._robot_body.linearVelocity = (0.0, 0.0)
            base._robot_body.angularVelocity = 0.0
            _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
            steps += 1
            vx, vy = base._cargo_body.linearVelocity
            if (vx * vx + vy * vy) ** 0.5 < SPEED_TOL or terminated or truncated:
                break
        final_x = float(base._cargo_body.position.x)
        coast = final_x - START_X
        # where the release had to happen so the crate stops at the dock centre
        release_x = 2.6 - coast
        # one control step of travel at this speed
        step_len = v0 * DT
        margin = DOCK_TOL / step_len if step_len > 0 else float("inf")
        print(f"{v0:>9.2f} {coast:>10.3f} {steps:>7d} {release_x:>13.3f} "
              f"{2 * DOCK_TOL:>11.2f} {margin:>15.1f}")
    env.close()


if __name__ == "__main__":
    main()
