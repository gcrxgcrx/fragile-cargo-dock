import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Ant-v4")
model = PPO.load("ant_rlzoo_baseline_seed0")  # 确保路径正确，不带 .zip

obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
print("Action sum:", action.sum())
print("Action min:", action.min(), "max:", action.max())
env.close()