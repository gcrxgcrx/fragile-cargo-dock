import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import numpy as np
import os
import torch

if __name__ == '__main__':
    ENV_ID = "Ant-v4"
    TOTAL_TIMESTEPS = 1_000_000
    N_ENVS = 8
    SEED = 0

    def make_env():
        def _init():
            env = gym.make(ENV_ID)
            return env
        return _init

    # 创建并行环境
    env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 调优的 PPO 参数（修正 activation_fn）
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            ortho_init=True,
            activation_fn=torch.nn.Tanh,    # 修改为类而不是字符串
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
        ),
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.0,
        learning_rate=3e-4,
        clip_range=0.2,
        max_grad_norm=0.5,
        vf_coef=0.5,
        verbose=1,
        seed=SEED,
        device="cpu",
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 保存模型和 VecNormalize 统计
    model.save("ant_rlzoo_baseline_seed0")
    env.save("vecnormalize_rlzoo_seed0.pkl")
    env.close()
    print("Training finished. Model and VecNormalize saved.")