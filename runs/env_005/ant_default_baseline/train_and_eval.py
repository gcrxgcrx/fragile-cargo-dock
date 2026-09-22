import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import torch
import time

if __name__ == '__main__':
    # ================== 训练参数 ==================
    ENV_ID = "Ant-v4"
    TOTAL_TIMESTEPS = 1_000_000
    N_ENVS = 4          # 单进程下用4个虚拟环境，平衡速度与稳定性
    SEED = 0

    # ================== 创建环境并归一化 ==================
    def make_env():
        def _init():
            env = gym.make(ENV_ID)
            return env
        return _init

    env = DummyVecEnv([make_env() for _ in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # ================== 调优的 PPO 模型 ==================
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            ortho_init=True,
            activation_fn=torch.nn.Tanh,
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

    # ================== 训练 ==================
    print("Starting training... (this may take 20-30 minutes)")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # 保存模型和归一化统计
    model.save("ant_rlzoo_baseline_seed0")
    env.save("vecnormalize_rlzoo_seed0.pkl")
    env.close()
    print("Training completed.\n")

    # ================== 评估 ==================
    print("Evaluating...")
    def make_eval_env():
        env = gym.make(ENV_ID, terminate_when_unhealthy=False)
        return env

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize.load("vecnormalize_rlzoo_seed0.pkl", eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load("ant_rlzoo_baseline_seed0")

    episode_rewards = []
    for ep in range(100):
        obs = eval_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward[0]
            done = done[0]
        episode_rewards.append(total_reward)
        print(f"Ep {ep+1:3d}: {total_reward:.2f}")

    rewards = np.array(episode_rewards)
    mean, std = np.mean(rewards), np.std(rewards)
    print(f"\nFinal Baseline (Ant-v4 RL Zoo tuned, seed 0):")
    print(f"Mean: {mean:.2f}, Std: {std:.2f}")
    print(f"Min: {np.min(rewards):.2f}, Max: {np.max(rewards):.2f}")

    np.savetxt("ant_rlzoo_baseline_100ep.csv", rewards, header="reward", comments="")
    print("Results saved to ant_rlzoo_baseline_100ep.csv")