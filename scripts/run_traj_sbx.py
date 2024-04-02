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
import mujoco_py
import cv2

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

import scripts.replay_traj_sbx as replay

def main(args):
    print("Running trajectories")

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
    if (os.path.exists(trajectory_dir) == False):
        os.makedirs(trajectory_dir)

        

    # run the agent x times
    for i in range(args.num_trajs):
        print("Running trial: " + str(i) + "...")
        # save images and state-action pairs
        images = []
        actions = []

        # make sure to save the exact state to replicate between trials and replay
        obs = eval_env.reset()
        state = eval_env.get_env_state()

        # save state as json
        with open(os.path.join(trajectory_dir, "state_" + str(i) + ".pickle" ), 'ab') as outfile:
            # print("state")
            # print(state)
            # obj_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                            #  old_state.act, old_state.udd_state)
            pickle.dump(state, outfile)

        img = eval_env.render(offscreen=True)
        # img = eval_env.render()
        images.append(img)

        # 500 timesteps
        for t in range(500):
            action, _ = model.predict(obs)
            obs, reward, done, info = eval_env.step(action)

            img = eval_env.render(offscreen=True)
            # img = eval_env.render()
            images.append(img)

            # pair = {"action": action}
            # print("pair:")
            # print(pair)
            # print(action)
            actions.append(action.tolist())

            if done:
                eval_env.reset()
                break

        # save each image
        for f, img in enumerate(images):
            cv2.imwrite(os.path.join(trajectory_dir, "button_press_" + str(i) + "_" + str(f) + ".png"), img)
        # imageio.mimsave(os.path.join(trajectory_dir, "button_press_" + str(i) + ".gif"), [np.array(img) for i, img in enumerate(images)], fps=30)

        # save images as mp4
        for f in range(len(images)):
            images[f] = cv2.imread(os.path.join(trajectory_dir, "button_press_" + str(i) + "_" + str(f) + ".png"))
        frame = cv2.imread(images[0])
        height, width, layers = frame.shape

        video = cv2.VideoWriter("button_press_" + str(i) + ".mp4", 0, 1, (width,height))

        for image in images:
            video.write(cv2.imread(image))

        cv2.destroyAllWindows()
        video.release()

        # delete the images
        for f in range(len(images)):
            os.remove(os.path.join(trajectory_dir, "button_press_" + str(i) + "_" + str(f) + ".png"))

        # save state-action pairs as json
        with open(os.path.join(trajectory_dir, "actions_" + str(i) + ".json"), 'w') as outfile:
            # for action in state_action_pairs:
            json.dump(actions, outfile)
        

    print("Done running trajectories!")

    if (args.run_replay):
        replay.main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", help="e.g., 0-1-1-2", type=str, default="0-0-0-0")
    parser.add_argument("--num-trajs", help="number of runs of this trained variant", type=int, default=4)
    parser.add_argument("--run-replay", help="if true, runs replay_traj_sbx.py to confirm replay works", type=bool, default=True)

    args = parser.parse_args()

    main(args)