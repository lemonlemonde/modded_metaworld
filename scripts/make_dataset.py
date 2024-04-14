import torch
from torch.utils.data import Dataset
import json
import numpy as np
import argparse
import os
# from utils import generate_synthetic_comparisons_commands, generate_noisyaugmented_synthetic_comparisons_commands, calc_and_set_global_vars

import pickle, bz2

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE



env = None
trajs = []
feature_vals = []
index_weights = []
features = ['height', 'velocity', 'distance_to_object', 'sum']

# default 500, will be ovewritten by num columnns in actions.json
NUM_TIMESTEPS = 500

# small dataset for testing, use GPT_augmented_dataset later for full dataset
greater_height_adjs = ["Move higher.", "Move taller.", "Move at a greater height."]
greater_velocity_adjs = ["Move faster.", "Move quicker.", "Move swifter.", "Move at a higher speed."]
greater_distance_adjs = ["Move further from the button.", "Move farther from the button.", "Move more distant from the button."]
greater_sum_adjs = ["Press the button better.", "Press the button more successfully."]
greater_adjs = [greater_height_adjs] + [greater_velocity_adjs] + [greater_distance_adjs] + [greater_sum_adjs]

lesser_height_adjs = ["Move lower.", "Move more down.", "Move at a lesser height."]
lesser_velocity_adjs = ["Move slower.", "Move more moderate.", "Move more sluggish.", "Move at a lower speed."]
lesser_distance_adjs = ["Move closer to the button.", "Move nearer to the button.", "Move more nearby to the button."]
lesser_sum_adjs = ["Press the button worse.", "Press the button not as well."]
lesser_adjs = [lesser_height_adjs] + [lesser_velocity_adjs] + [lesser_distance_adjs] + [lesser_sum_adjs]

# calculates the feature values for all trajectories based on compute_reward_v2()
def calc_feature_vals():
    print("*****Calculating feature vals*****")
    f_vals = []
    flag = False
    for trajIndex, traj in enumerate(trajs):
        avg_sum_vals = []
        height_vals = []
        velocity_vals = []
        distance_to_obj_vals = []
        for step in range(0, NUM_TIMESTEPS):
            # compute_reward_v2(action, observation)
            # traj = [observation(shape 39), action(4)]
            reward, avg_sum, tcp_height, tcp_vel, tcp_to_obj, env_state = env.compute_reward_v2(traj[step][39:], traj[step][:39])
            
            print("computed reward: ")
            print(avg_sum)
            print(tcp_height)
            print(tcp_vel)
            print(tcp_to_obj)
            print("------")
            if (trajIndex == 24 and step == 300):
                # print("observation: ")
                # print(traj[step][:39])
                # print(traj[step][:39].shape)
                # print("action:")
                # print(traj[step][39:])
                # print(traj[step][39:].shape)

                
                flag = True
            
            avg_sum_vals.append(avg_sum)
            height_vals.append(tcp_height)
            velocity_vals.append(tcp_vel)
            distance_to_obj_vals.append(tcp_to_obj)
        f_vals.append([height_vals] + [velocity_vals] + [distance_to_obj_vals] + [avg_sum_vals])

    return f_vals

# reads states and actions from files and returns them as a list of trajectories (observations + actions)
def form_trajectories():
    print("*****Forming trajectories = (observations + actions)*****")
    # trajectories/0-0-0-0/actions_0.json
    # trajectories/0-0-0-0/states_0.pickle
    temp = []
    flag = True

    # e.g., 0-0-0-0
    for variant in index_weights:
        # 0 to 3
        for trial in range(0, 4):
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            dir = os.path.join(cur_dir, "../trajectories", variant)

            # load saved states and actions
            with open(os.path.join(dir, "actions_" + str(trial) + ".json"), 'r') as openfile:
                actions = json.load(openfile)
            with open(os.path.join(dir, "states_" + str(trial) + ".pickle"), 'rb') as openfile:
                states = pickle.load(openfile)

            # Get the observations
            observations = []
            NUM_TIMESTEPS = len(actions)
            for step in range(0, NUM_TIMESTEPS):
                env.set_env_state(states[step])
                obs = env._get_obs()
                observations.append(obs)
                # env.compute_reward_v2(actions[step], obs)
            

            observations = np.array(observations)
            actions = np.array(actions)
            traj = np.concatenate((observations, actions), axis=-1)
            temp.append(traj)
            if (not flag):
                print("observation: ")
                print(observations)
                print(observations.shape)
                print("actions: " )
                print(actions)
                print(actions.shape)
                print("traj:")
                print(traj)
                print(traj.shape)
                flag = True

    # print("example temp[0]")
    # print(temp[0])
    # print(temp[0].shape)
    # print(temp[0][400][0])
    print("NUM_TIMESTEPS: " + str(NUM_TIMESTEPS))
    return temp

# i, j = indices of traj in order of ["0-0-0-0", "0-0-0-1", ... "2-2-2-2"]
# feature = {'height', 'velocity', 'distance_to_object', 'sum'}
# noisy = if True, 1% chance of opposite comparison
# returns string of language feedback of traj i to traj j
def generate_synthetic_lang_feedback(i, j, feature, noisy=False):
    # 1 percent chance of getting incorrect comparison
    prob = 0.01

    # get the weights of the two trajectories
    # cur_dir = os.path.dirname(os.path.abspath(__file__))
    # variant_path_i = os.path.abspath(os.path.join(cur_dir, "..", "training_configs/sawyer_button_press_v2/variants/", index_weights[i]))
    # variant_path_j = os.path.abspath(os.path.join(cur_dir, "..", "training_configs/sawyer_button_press_v2/variants/", index_weights[j]))
    # with open(os.path.join(variant_path_i, "variant.json"), 'r') as openfile:
    #     variant_file_i = json.load(openfile)
    # with open(os.path.join(variant_path_j, "variant.json"), 'r') as openfile:
    #     variant_file_j = json.load(openfile)
    # weights_i = variant_file_i["eval_environment_kwargs"]["weights"]
    # weights_j = variant_file_j["eval_environment_kwargs"]["weights"]

    # get the average feature values of the two trajectories
    feature_vals_i = feature_vals[i]
    feature_vals_j = feature_vals[j]
    
    # index of feature
    f = features.index(feature)

    avg_i = np.mean(feature_vals_i[f])
    avg_j = np.mean(feature_vals_j[f])

    # check what language feedback to give
    if (avg_i > avg_j):
        # i is greater
        if (noisy and np.random.rand() < prob):
            # by prob chance, return the opposite comparison\
            # j is greater
            return greater_adjs[f][np.random.randint(len(greater_adjs[f]))]
        else:
            # i is greater
            return lesser_adjs[f][np.random.randint(len(lesser_adjs[f]))]
    else:
        # j is greater
        if (noisy and np.random.rand() < prob):
            # by prob chance, return the opposite comparison
            # i is greater
            return lesser_adjs[f][np.random.randint(len(lesser_adjs[f]))]
        else:
            # j is greater
            return greater_adjs[f][np.random.randint(len(greater_adjs[f]))]

def get_comparisons(i, j, noisy=False):
    out = []
    for feature in ['height', 'velocity', 'distance_to_object', 'sum']:
        out.append(generate_synthetic_lang_feedback(i, j, feature, noisy=noisy))
    return out


def initialize_globals():
    global env, trajs, feature_vals, index_weights

    print("*****Initializing global variables*****")

    # init index_weights
    # Get all possible weights "0-0-0-0", "0-0-0-1", ... "2-2-2-2
    for height in range(3):
        for vel in range(3):
            for dist in range(3):
                for sum in range(3):
                    index_weights.append(str(height) + "-" + str(vel) + "-" + str(dist) + "-" + str(sum))


    # init env
    button_press_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["button-press-v2-goal-observable"]
    env = button_press_goal_observable_cls()
    env._freeze_rand_vec = False

    # init trajs and feature_vals
    trajs = form_trajectories()
    feature_vals = calc_feature_vals()

    # print("trajs:")
    # print(trajs)
    # print("feature vals:")
    # print(feature_vals)


# generate traj a's, traj b's, and comparisons from a --> b
def generate_dataset(noisy=False, id_mapping=False, all_pairs=True):
    print("*****Generating Dataset*****")
    # a --> b with language feedback comps
    dataset_traj_as = []
    dataset_traj_bs = []
    dataset_comps = []

    if (all_pairs):
        print("**generating all pairs...***")
        # all pairs where trajs = (0-0-0-0, ..., 2-2-2-2) each with 4 trials, as specified for run_traj_sbx.py
        for i in range(0, len(trajs)):
            for j in range(i + 1, len(trajs)):
                comps = get_comparisons(i, j, noisy=noisy)
                flipped_comps = get_comparisons(j, i, noisy=noisy)

                for c in comps:
                    if (id_mapping):
                        dataset_traj_as.append(i)
                        dataset_traj_bs.append(j)
                        dataset_comps.append(c)
                    else:
                        dataset_traj_as.append(trajs[i])
                        dataset_traj_bs.append(trajs[j])
                        dataset_comps.append(c)
                for fc in flipped_comps:
                    if (id_mapping):
                        dataset_traj_as.append(j)
                        dataset_traj_bs.append(i)
                        dataset_comps.append(fc)
                    else:
                        dataset_traj_as.append(trajs[j])
                        dataset_traj_bs.append(trajs[i])
                        dataset_comps.append(fc)
                    
    else:
        print("***generating random pairs...***")
        # random pairs
        for n in range(0, len(trajs)):
            i = 0
            j = 0
            while i == j:
                i = np.random.randint(len(trajs))
                j = np.random.randint(len(trajs))

            comps = get_comparisons(i, j, noisy=noisy)
            flipped_comps = get_comparisons(j, i, noisy=noisy)

            for c in comps:
                if (id_mapping):
                    dataset_traj_as.append(i)
                    dataset_traj_bs.append(j)
                    dataset_comps.append(c)
                else:
                    dataset_traj_as.append(trajs[i])
                    dataset_traj_bs.append(trajs[j])
                    dataset_comps.append(c)
            for fc in flipped_comps:
                if (id_mapping):
                    dataset_traj_as.append(j)
                    dataset_traj_bs.append(i)
                    dataset_comps.append(fc)
                else:
                    dataset_traj_as.append(trajs[j])
                    dataset_traj_bs.append(trajs[i])
                    dataset_comps.append(fc)

    return dataset_traj_as, dataset_traj_bs, dataset_comps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    # parser.add_argument('--policy-dir', type=str, default='', help='')
    # parser.add_argument('--output-dir', type=str, default='', help='')
    # parser.add_argument('--dataset-size', type=int, default=1000, help='')
    # parser.add_argument('--trajs-per-policy', type=int, default=5, help='')
    # parser.add_argument('--trajs-per-expert-policy', type=int, default=5, help='')
    # parser.add_argument('--val-split', type=float, default=0.1, help='')
    # parser.add_argument('--with-videos', action="store_true", help='')
    # parser.add_argument('--use-img-obs', action="store_true", help='')
    parser.add_argument('--noise-augmentation', type=bool, default=False, help='')
    parser.add_argument('--id-mapping', action="store_true", help='')
    parser.add_argument('--all-pairs', action="store_true", help='')
    parser.add_argument('--seed', type=int, default=0, help='')

    args = parser.parse_args()

    # policy_dir = args.policy_dir
    # output_dir = args.output_dir
    # dataset_size = args.dataset_size
    # trajs_per_policy = args.trajs_per_policy
    # trajs_per_expert_policy = args.trajs_per_expert_policy
    # val_split = args.val_split
    # with_videos = args.with_videos
    noise_augmentation = args.noise_augmentation
    id_mapping = args.id_mapping
    all_pairs = args.all_pairs
    seed = args.seed

    np.random.seed(seed)
    initialize_globals()
    dataset_traj_as, dataset_traj_bs, dataset_comps = generate_dataset(noisy=noise_augmentation, id_mapping=id_mapping, all_pairs=all_pairs)