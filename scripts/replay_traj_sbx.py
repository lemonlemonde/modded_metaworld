import os
import pickle
import time
import numpy as np
# from sbx import SAC
from sbx import SAC
from stable_baselines3.common.save_util import load_from_zip_file
import imageio
import argparse
import json

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

def main(args):    
    # replay the trajectory
    print("Now replay the trajectory:")

    # directory
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    trajectory_dir = os.path.join(cur_dir, "../trajectories", args.variant)

    # get the saved state to replay
    with open(os.path.join(trajectory_dir, "state.pickle"), 'rb') as openfile:
        state = pickle.load(openfile)

    # Init the environment
    button_press_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["button-press-v2-goal-observable"]
    eval_env = button_press_goal_observable_cls()
    eval_env._freeze_rand_vec = False

    # Opening JSON file for weight variants
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    variant_path = os.path.abspath(os.path.join(cur_dir, "..", "training_configs/sawyer_button_press_v2/variants/", args.variant))
    filename = "variant.json"

    with open(os.path.join(variant_path, filename), 'r') as openfile:
        variantFile = json.load(openfile)

    eval_env.set_variant(variantFile["eval_environment_kwargs"])

    
    # get the state-action pairs
    for i in range(args.num_trajs):
        print("replaying trajectory " + str(i))
        eval_env.reset()
        eval_env.set_env_state(state)
        images = []

        with open(os.path.join(trajectory_dir, "actions_" + str(i) + ".json"), 'r') as openfile:
            actions = json.load(openfile)
            # replay the trajectory
            # action = pair["action"]
            # state = pair["state"]
            # eval_env.set_env_state(obs)
            for action in actions:
                eval_env.step(np.array(action))
                img = eval_env.render(offscreen=True)
                images.append(img)

        imageio.mimsave(os.path.join(trajectory_dir, "replay_" + str(i) + ".gif"), [np.array(img) for i, img in enumerate(images)], fps=30)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", help="e.g., 0-1-1-2", type=str, default="0-0-0-0")
    parser.add_argument("--num-trajs", help="number of runs of this trained variant", type=int, default=4)

    args = parser.parse_args()

    main(args)