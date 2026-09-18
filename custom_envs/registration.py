"""Centralised Gymnasium registration for repository-local custom environments.

Why this module exists
----------------------
Environments must be registered *before* ``gym.make`` is called. With
``SubprocVecEnv`` every worker process re-imports ``training.train_sb3_wrapper``,
so the registration has to happen at module import time in that entry point.
Previously the ``register(...)`` calls were duplicated and hard-coded (together
with an absolute Windows path) in both entry points.

Registering here instead gives one place to add an environment, keeps the
import path relative to the repository, and deliberately imports *no* third
party simulator: entry points are strings, so a missing simulator only surfaces
when that particular environment is actually instantiated.
"""

from __future__ import annotations

from gymnasium.envs.registration import register, registry

#: All custom environments shipped with this repository.
CUSTOM_ENVS = (
    {
        "id": "FragileCargoDock-v0",
        "entry_point": "custom_envs.fragile_cargo_dock_env:FragileCargoDockEnv",
        "max_episode_steps": 400,
    },
)


def register_custom_envs() -> None:
    """Register every custom environment that is not registered yet."""
    for spec in CUSTOM_ENVS:
        env_id = spec["id"]
        if env_id in registry:
            continue
        register(**spec)


register_custom_envs()
