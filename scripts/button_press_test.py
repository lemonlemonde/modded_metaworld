import time
import metaworld
import random


# print(metaworld.ML1.ENV_NAMES)  # Check out the available environments
# button envis:
    # 'button-press-topdown-v2'
    # 'button-press-topdown-wall-v2'
    # 'button-press-v2'
    # 'button-press-wall-v2'
    # 'coffee-button-v2'


SEED = 0

# metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_button_press_v2.py
ml1 = metaworld.ML1('button-press-v2', seed=SEED) # Construct the benchmark, sampling tasks
env = ml1.train_classes['button-press-v2']()  # Create an environment with task `sawyer-button-press-v2`

task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

obs = env.reset()  # Reset environment

for _ in range(200):
    a = env.action_space.sample()  # Sample an action
    obs, reward, done, info, features = env.step(a)  # Step the environment with the sampled random action

    # print("dimensions of obs: ", obs.shape) # (39,)
    env.render()  # Render the environment

    # TODO: 4 features (including reward(avg) itself)
    # avg of the following (ground truth reward)
    # height
    # velocity
    # distance to button
    
    print("---------------")
    print("timestep: ", _)
    print("reward: ", reward)
    print("avg sum: ", features["avg_sum"])
    print("tcp_height: ", features["tcp_height"])
    print("tcp_vel: ", features["tcp_vel"])
    print("tcp_to_obj: ", features["tcp_to_obj"])

    time.sleep(0.05)
  
# rewards they have:
    # 'success': distance between object and target (0.02 is the threshold for success)
    # 'near_object': whether or not near the button
    # 'grasp_success': if end effector is open or closed
    # 'grasp_reward': reward for having end effector open or closed
    # 'in_place_reward': button is indeed pressed
    # 'obj_to_target': distance between object and target (not dependent on threshold 0.02)
    # 'unscaled_reward': reward = 2 * reward_utils.hamacher_product(tcp_closed, near_button)


# TODO: check Jeremy's RL policy training, and do similar
# TODO: rollouts with randomized weights for each of the 4 reward funcs