import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np

ENV_ID = "Ant-v4"
MODEL_PATH = "ant_rlzoo_baseline_seed0"
VECNORM_PATH = "vecnormalize_rlzoo_seed0.pkl"
N_EPISODES = 100

# 创建单环境用于评估（注意要包装 VecNormalize）
def make_env():
    env = gym.make(ENV_ID, terminate_when_unhealthy=False)
    return env

env = DummyVecEnv([make_env])
env = VecNormalize.load(VECNORM_PATH, env)
# 评估模式：不更新统计量，且使用原始奖励（不要归一化奖励）
env.training = False
env.norm_reward = False

model = PPO.load(MODEL_PATH)

episode_rewards = []

for ep in range(N_EPISODES):
    obs = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # 由于 env 是向量化的，奖励是一个数组，取第一个元素
        total_reward += reward[0]
        steps += 1
        done = done[0]  # 单环境所以取第一个

    episode_rewards.append(total_reward)
    print(f"Episode {ep+1:3d}: reward = {total_reward:.2f}, steps = {steps}")

rewards = np.array(episode_rewards)
mean = np.mean(rewards)
std = np.std(rewards)
print(f"\nMean: {mean:.2f}, Std: {std:.2f}")
print(f"Min: {np.min(rewards):.2f}, Max: {np.max(rewards):.2f}, Median: {np.median(rewards):.2f}")
np.savetxt("ant_rlzoo_baseline_100ep.csv", rewards, header="reward", comments="")