import os
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

    # Load the trained agent
    train_dir = os.path.join(cur_dir, "..", "training_results", args.variant)
    print("train_dir: " + train_dir)
    data, params, pytorch_variables = load_from_zip_file(
        os.path.join(train_dir, "sac_button_press_v2_goal_observable"),
        device="auto",
        )
    model = SAC("MlpPolicy", eval_env, verbose=1, batch_size=500)
    model.set_parameters(params, exact_match=True, device="auto")


    

    # directory
    trajectory_dir = os.path.join(cur_dir, "../trajectories", args.variant)

    # run the agent x times
    for i in range(args.num_trajs):
        # save images and state-action pairs
        images = []
        state_action_pairs = []

        obs = eval_env.reset()
        img = eval_env.render(offscreen=True)
        # 500 timesteps
        for t in range(500):
            action, _ = model.predict(obs)
            obs, reward, done, info = eval_env.step(action)

            # sleep
            time.sleep(0.1)

            pair = (action, info["env_state"])
            print("pair:")
            print(pair)
            state_action_pairs.append(pair)

            if done:
                eval_env.reset()
                break
            if i == 0:
                images.append(img)
                img = eval_env.render(offscreen=True)

        imageio.mimsave(os.path.join(trajectory_dir, "button_press_" + i + ".gif"), [np.array(img) for i, img in enumerate(images)], fps=30)
        # save state-action pairs as json
        with open(os.path.join(trajectory_dir, "state_action_pairs_" + i ".json"), 'w') as outfile:
            json.dump(state_action_pairs, outfile)




    print("Done!")

    print("Now replay the trajectory:")

    # replay the trajectory
    # get the state-action pairs
    for i in range(args.num_trajs):
        with open(os.path.join(trajectory_dir, "state_action_pairs_" + i + ".json"), 'r') as openfile:
            state_action_pairs = json.load(openfile)

        eval_env.reset()

        # replay the trajectory
        for pair in state_action_pairs:
            action = pair[0]
            obs = pair[1]
            eval_env.set_env_state(obs)
            eval_env.step(action)
            eval_env.render()

            # sleep
            time.sleep(0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", help="e.g., 0-1-1-2", type=str, default="0-0-0-0")
    parser.add_argument("--num-trajs", help="number of runs of this trained variant", type=int, default=4)

    args = parser.parse_args()

    main(args)