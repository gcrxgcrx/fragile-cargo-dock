import gymnasium as gym
from stable_baselines3 import PPO

ENV_ID = "Ant-v4"
MODEL_PATH = "ant_rlzoo_baseline_seed0"

env = gym.make(ENV_ID, terminate_when_unhealthy=False)
model = PPO.load(MODEL_PATH)

for ep in range(3):
    obs, _ = env.reset(seed=42 + ep)
    done = False
    total_reward = 0
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
    print(f"Ep {ep}: steps={steps}, reward={total_reward:.1f}")