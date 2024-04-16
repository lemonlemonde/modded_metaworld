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

# small language feedback dataset for testing, use GPT_augmented_dataset later for full dataset
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


# reads states and actions from files and returns them as a list of trajectories
    # every trajectory = (observations + actions) = (500, 43) for 500 timesteps
    # observations = (500, 39) for 500 timesteps
    # actions = (500, 4) for 500 timesteps
# also calculates feature values for each trajectory and returns them
def form_trajectories():
    print("\t-->>-- Forming Trajectories --<<--")
    
    formatted_trajs = []
    f_vals = []

    # e.g., 0-0-0-0
    for variant in index_weights:
        # 0 to 3
        for trial in range(0, 4):
            avg_sum_vals = []
            height_vals = []
            velocity_vals = []
            distance_to_obj_vals = []

            cur_dir = os.path.dirname(os.path.abspath(__file__))
            dir = os.path.join(cur_dir, "../trajectories", variant)
            # trajectories/0-0-0-0/actions_0.json
            # trajectories/0-0-0-0/states_0.pickle

            # load saved states and actions
            with open(os.path.join(dir, "actions_" + str(trial) + ".json"), 'r') as openfile:
                actions = json.load(openfile)
            with open(os.path.join(dir, "states_" + str(trial) + ".pickle"), 'rb') as openfile:
                states = pickle.load(openfile)

            # Get the observations
            observations = []
            NUM_TIMESTEPS = len(actions)
            for step in range(0, NUM_TIMESTEPS):
                # Get obs and rewards
                env.set_env_state(states[step])
                obs = env._get_obs()
                observations.append(obs)
                reward, avg_sum, tcp_height, tcp_vel, tcp_to_obj, env_state = env.compute_reward_v2(actions[step], obs)
                
                avg_sum_vals.append(avg_sum)
                height_vals.append(tcp_height)
                velocity_vals.append(tcp_vel)
                distance_to_obj_vals.append(tcp_to_obj)
            
            # Add formatted trajectory
            observations = np.array(observations)
            actions = np.array(actions)
            traj = np.concatenate((observations, actions), axis=-1)
            formatted_trajs.append(traj)

            # Add feature values
            f_vals.append([height_vals] + [velocity_vals] + [distance_to_obj_vals] + [avg_sum_vals])

    # print("NUM_TIMESTEPS: " + str(NUM_TIMESTEPS))
    # print("fvals shape: ")
    # print(np.array(f_vals).shape)
    return formatted_trajs, f_vals

# modifies global variables greater_adjs and lesser_adjs with gpt augmented dataset json file
def get_gpt_dataset():
    global greater_adjs, lesser_adjs

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(cur_dir, "../dataset")
    with open(os.path.join(dir, "gpt_augmented_dataset_metaworld.json"), 'r') as openfile:
        gpt_dataset = json.load(openfile)
    
    greater_adjs = gpt_dataset["greater_height"] + gpt_dataset["greater_velocity"] + gpt_dataset["greater_distance"] + gpt_dataset["greater_sum"]
    lesser_adjs = gpt_dataset["lesser_height"] + gpt_dataset["lesser_velocity"] + gpt_dataset["lesser_distance"] + gpt_dataset["lesser_sum"]

# i, j = indices of traj in order of ["0-0-0-0", "0-0-0-1", ... "2-2-2-2"], where each variant has 4 trials
# feature = {'height', 'velocity', 'distance_to_object', 'sum'}
# noisy = if True, 1% chance of opposite comparison
# returns string of language feedback of traj i --> traj j
def generate_synthetic_lang_feedback(i, j, feature, use_gpt_dataset, noisy=False):
    # 1 percent chance of getting incorrect comparison
    prob = 0.01

    # get the average feature values of the two trajectories
    feature_vals_i = feature_vals[i]
    feature_vals_j = feature_vals[j]
    
    # index of feature in order of 
        # {'height', 'velocity', 'distance_to_object', 'sum'}
    f = features.index(feature)

    # average value of feature over 500 timesteps
    avg_i = np.mean(feature_vals_i[f])
    avg_j = np.mean(feature_vals_j[f])

    if (use_gpt_dataset):
        # this modifies greater_adjs and lesser_adjs
        get_gpt_dataset()

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
            print("some adj: " + str(lesser_adjs[f][np.random.randint(len(lesser_adjs[f]))]))
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

    print("\t-->>-- Initializing Global Variables! --<<--")

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
    trajs, feature_vals = form_trajectories()


# generate traj a's, traj b's, and comparisons from a --> b
def generate_dataset(noisy=False, id_mapping=False, all_pairs=True):
    
    # a --> b with language feedback comps
    dataset_traj_as = []
    dataset_traj_bs = []
    dataset_comps = []

    if (all_pairs):
        print("\t-->>-- Generating Dataset (all pairs)! --<<--")
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
        print("\t-->>-- Generating Dataset (random pairs)! --<<--")
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
    print("-->>-->>-- Making dataset! --<<--<<--")
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
    parser.add_argument('--use-gpt-dataset', type=bool, default=True, help='')

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

    # save dataset
        # 324 trajs (81 variants x 4 trials each)
        # 500 timesteps
        # 418608 is from:
            # (81 variants x 4 trials each) = 324
            # 324 choose 2 (i-->j) = 52326
            # 52326 x 4 features = 209304
            # 209304 x 2 for flipped order (j-->i) = 418608
        # dataset_traj_as.npy: 
            # contains indices or trajectories (observations + actions)
            # e.g., [0, 0, 0, 0, 1, ..., 322, 322, 322, 322]
        # dataset_traj_bs.npy: 
            # contains indices or trajectories (observations + actions)
            # e.g., [1, 1, 1, 1, 2, ..., 323, 323, 323, 323]
        # dataset_comps.npy: 
            # contains language feedback phrases to map trajectory a's --> trajectory b's for every feature ['height', 'velocity', 'distance_to_object', 'sum'], possibly with error if noisy
            # e.g., ["Move more down", "Move faster", "Move closer to the button", "Move better", ..., ...]

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(cur_dir, "../dataset")
    if (os.path.exists(dir) == False):
            os.makedirs(dir)
    np.save(os.path.join(dir, "dataset_traj_as.npy"), np.array(dataset_traj_as))
    np.save(os.path.join(dir, "dataset_traj_bs.npy"), np.array(dataset_traj_bs))
    np.save(os.path.join(dir, "dataset_comps.npy"), np.array(dataset_comps))

    print("-->>-->>-- Done making dataset!!! --<<--<<--")

    # open np files and check
    print("-->>-->>-- Checking saved files... --<<--<<--")
    print("dataset_traj_as:")
    traj_as = np.load(os.path.join(dir, "dataset_traj_as.npy"))
    print(traj_as)
    print(traj_as.shape)
    # for i in range(0, 17):
    #     print(traj_as[i])

    print("dataset_traj_bs:")
    traj_bs = np.load(os.path.join(dir, "dataset_traj_bs.npy"))
    print(traj_bs)
    print(traj_bs.shape)
    # for i in range(0, 17):
    #     print(traj_bs[i])

    print("dataset_comps:")
    comps = np.load(os.path.join(dir, "dataset_comps.npy"))
    print(comps)
    print(comps.shape)
    print("-->>-->>-- Check ^^^ everything looks okay?? --<<--<<--")