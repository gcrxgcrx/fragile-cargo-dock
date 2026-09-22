import gymnasium as gym
from stable_baselines3 import PPO
import time

ENV_ID = "Ant-v4"
MODEL_PATH = "ant_default_baseline_seed0"   # 改成你的实际路径

env = gym.make(ENV_ID, render_mode="human", terminate_when_unhealthy=False)
model = PPO.load(MODEL_PATH)

obs, _ = env.reset(seed=42)
done = False
truncated = False

# 获取蚂蚁身体的 ID（通常 torso 是第一个身体）
torso_id = env.unwrapped.model.body("torso").id   # 对于 Ant-v4，躯干名称为 "torso"

while not (done or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)

    # 更新相机位置，使其跟随蚂蚁的躯干
    if hasattr(env.unwrapped, "mujoco_renderer"):
        # 对于较新版本的 gymnasium，使用 mujoco_renderer 的 camera 方法
        viewer = env.unwrapped.mujoco_renderer.viewer
        if viewer is not None:
            viewer.cam.lookat[:] = env.unwrapped.data.xpos[torso_id].copy()
            viewer.cam.distance = 5.0    # 可根据需要调整距离
            viewer.cam.elevation = -30   # 俯仰角
            viewer.cam.azimuth = 90      # 水平角

    #time.sleep(0.05)   # 减慢速度，便于观察

print(f"Episode reward: {reward:.2f}")
env.close()