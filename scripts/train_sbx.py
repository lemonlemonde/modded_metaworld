import numpy as np
from sbx import SAC
from stable_baselines3.common.callbacks import EvalCallback
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
import imageio

button_press_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["button-press-v2-goal-observable"]
button_press_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN["button-press-v2-goal-hidden"]

# Create the environment
env = button_press_goal_observable_cls()
env._freeze_rand_vec = False

# Separate evaluation env
eval_env = button_press_goal_observable_cls(render_mode='rgb_array')
eval_env._freeze_rand_vec = False

# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=100_000,
                             deterministic=True, render=False)

model = SAC("MlpPolicy", env, verbose=1, batch_size=500)
model.learn(total_timesteps=2_500_000, callback=eval_callback)
print("Training done!")

model.save("sac_button_press_goal_observable")
# model = SAC.load("sac_pick_place_v2_goal_observable", env=eval_env)

images = []
obs, _ = eval_env.reset()
img = eval_env.render()
for i in range(500):
    images.append(img)
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = eval_env.step(action)
    print(info['is_success'], info['obj_to_target'])
    img = eval_env.render()
    if done:
        break

imageio.mimsave("button_press.gif", [np.array(img) for i, img in enumerate(images)], fps=30)
