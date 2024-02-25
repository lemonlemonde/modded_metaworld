from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)
                            # these are ordered dicts where the key : value
                            # is env_name : env_constructor

import numpy as np
from sbx import SAC
from stable_baselines3.common.callbacks import EvalCallback

button_press_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["button-press-v2-goal-observable"]
button_press_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN["button-press-v2-goal-hidden"]

# Create the environment
env = button_press_goal_observable_cls(seed=42, render_mode=None)

# Separate evaluation env
eval_env = button_press_goal_observable_cls(seed=42, render_mode=None)
# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=10_000,
                             deterministic=True, render=False)

model = SAC("MlpPolicy", env, verbose=1, batch_size=500)
model.learn(total_timesteps=250_000, callback=eval_callback)
print("Training done!")
