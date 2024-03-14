import os
import numpy as np
# from sbx import SAC
from stable_baselines3 import SAC
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
    data, params, pytorch_variables = load_from_zip_file(
        os.path.join(train_dir, "sac_button_press_v2_goal_observable"),
        device="auto",
        )
    model = SAC("MlpPolicy", eval_env, verbose=1, batch_size=500)
    model.set_parameters(params, exact_match=True, device="auto")


    # Evaluate the agent
    images = []
    success_count = 0
    for i in range(10):
        obs = eval_env.reset()
        img = eval_env.render(offscreen=True)
        for t in range(500):
            action, _ = model.predict(obs)
            obs, reward, done, info = eval_env.step(action)
            # print(info['is_success'], info['obj_to_target'], reward)
            if info['is_success']:
                success_count += 1
                break
            if done:
                eval_env.reset()
                break
            if i == 0:
                images.append(img)
                img = eval_env.render(offscreen=True)

    print("Success rate: ", success_count / 10)
    imageio.mimsave(os.path.join(train_dir, "button_press_eval.gif"), [np.array(img) for i, img in enumerate(images)], fps=30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", help="e.g., 0-1-1-2", type=str, default="0-0-0-0")

    args = parser.parse_args()

    main(args)