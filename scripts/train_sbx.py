import argparse
import json
import os
import numpy as np
from sbx import SAC
from stable_baselines3.common.callbacks import EvalCallback
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
import imageio
import sys



# declare main function
def main(args, save_path):

    button_press_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["button-press-v2-goal-observable"]
    button_press_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN["button-press-v2-goal-hidden"]

    # Opening JSON file for weight variants
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    variant_path = os.path.abspath(os.path.join(cur_dir, "..", "training_configs/sawyer_button_press_v2/variants/", args.variant))
    filename = "variant.json"

    with open(os.path.join(variant_path, filename), 'r') as openfile:
        variantFile = json.load(openfile)

    variant_expl = variantFile["expl_environment_kwargs"]
    variant_eval = variantFile["eval_environment_kwargs"]


     # Create the environment
    env = button_press_goal_observable_cls()
    env._freeze_rand_vec = False
    env.set_variant(variant_expl)

    # Separate evaluation env
    eval_env = button_press_goal_observable_cls(render_mode='rgb_array')
    eval_env._freeze_rand_vec = False
    eval_env.set_variant(variant_eval)

    # Use deterministic actions for evaluation

    model = SAC("MlpPolicy", env, verbose=1, batch_size=500)
    model.learn(total_timesteps=300_000) # 250_000
    print("Training done!")

    model.save(save_path + "/sac_button_press_v2_goal_observable")
    model = SAC.load(save_path + "/sac_button_press_v2_goal_observable", env=eval_env)

    images = []
    obs, _ = eval_env.reset()
    img = eval_env.render()
    success_count = 0
    for i in range(10):
        for t in range(500):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = eval_env.step(action)
            # print(info['is_success'], info['obj_to_target'], reward)
            if i == 0:
                images.append(img)
                img = eval_env.render()
            if info['is_success']:
                success_count += 1
                break

    print("Success rate: ", success_count / 10)
    imageio.mimsave(save_path + "/button_press.gif", [np.array(img) for i, img in enumerate(images)], fps=30)

# call main function
if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", help="e.g., 0-1-1-2", type=str, default="0-0-0-0")

    args = parser.parse_args()

    # change stdout to a file
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.abspath(os.path.join(cur_dir, "..", "training_results", args.variant))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = open(save_path + '/out.txt', 'w')
    sys.stdout = f
    sys.stderr = f

    print("args variant: " + args.variant)
    
    main(args, save_path)

    # change it back!
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    f.close()