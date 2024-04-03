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
import cv2

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

def main(args):    
    # replay the trajectory
    print("Replaying trajectories")

    # directory
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    trajectory_dir = os.path.join(cur_dir, "../trajectories", args.variant)

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
        print("Replaying trajectory " + str(i) + "...")
 
        # get the saved state json to replay
        with open(os.path.join(trajectory_dir, "state_" + str(i) + ".json"), 'r') as openfile:
            state = json.load(openfile)
        
        # state = np.load(os.path.join(trajectory_dir, "state_" + str(i) + ".npy"))
            eval_env.reset()
            eval_env.set_env_state(state)
        images = []

        # get the saved actions to replay
        # actions = np.load(os.path.join(trajectory_dir, "actions_" + str(i) + ".npy"))
        with open(os.path.join(trajectory_dir, "actions_" + str(i) + ".json"), 'r') as openfile:
            actions = json.load(openfile)
            for action in actions:
                eval_env.step(np.array(action))
                img = eval_env.render(offscreen=True)
                images.append(img)

        # save images as npy
        image_dir = os.path.join(trajectory_dir, "replay_images_" + str(i))
        np.save(os.path.join(image_dir, "replay_images_" + str(i) + ".npy"), images)

        # save images as pngs to make a video
        for f, img in images:
            cv2.imwrite(os.path.join(image_dir, "img_" + str(i) + "_" + str(f) + ".png"), img)


        images = [img for img in os.listdir(image_dir) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(image_dir, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter("replay_" + str(i) + ".mp4", cv2.VideoWriter_fourcc(*'XVID'), 30, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_dir, image)))

        cv2.destroyAllWindows()
        video.release()

        # deletee images
        # for f, img in images:
        #     os.remove(os.path.join(image_dir, "img_" + str(i) + "_" + str(f) + ".png"))


    print("Done replaying trajectories!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", help="e.g., 0-1-1-2", type=str, default="0-0-0-0")
    parser.add_argument("--num-trajs", help="number of runs of this trained variant", type=int, default=4)

    args = parser.parse_args()

    main(args)