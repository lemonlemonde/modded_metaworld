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
trajs_train = []
trajs_test = []
trajs_val = []

feature_vals = []
feature_vals_train = []
feature_vals_test = []
feature_vals_val = []

index_weights = []
features = ['height', 'velocity', 'distance_to_object', 'sum']

# default 500, will be ovewritten by num columnns in actions.json
NUM_TIMESTEPS = 500

# small language feedback dataset for testing, use GPT_augmented_dataset later for full dataset
greater_height_adjs = ["Move higher.", "Move taller.", "Move at a greater height."]
greater_velocity_adjs = ["Move faster.", "Move quicker.", "Move swifter.", "Move at a higher speed."]
greater_distance_adjs = ["Move further from the button.", "Move farther from the button.", "Move more distant from the button."]
# greater_sum_adjs = ["Press the button better.", "Press the button more successfully."]
greater_adjs = [greater_height_adjs] + [greater_velocity_adjs] + [greater_distance_adjs]

lesser_height_adjs = ["Move lower.", "Move more down.", "Move at a lesser height."]
lesser_velocity_adjs = ["Move slower.", "Move more moderate.", "Move more sluggish.", "Move at a lower speed."]
lesser_distance_adjs = ["Move closer to the button.", "Move nearer to the button.", "Move more nearby to the button."]
lesser_sum_adjs = ["Press the button worse.", "Press the button not as well."]
lesser_adjs = [lesser_height_adjs] + [lesser_velocity_adjs] + [lesser_distance_adjs]

all_adjs = []

# sets global variables trajs_train, trajs_test, trajs_val, feature_vals_train, feature_vals_test, feature_vals_val
    # splits trajs + feature_vals into train, val, test
    # also splits greater_adjs and lesser_adjs into train, val, test
def split_dataset(split_train, split_val, split_test, size, split_lang_train, split_lang_test, split_lang_val, size_lang):
    global feature_vals
    global greater_adjs, lesser_adjs
    global trajs_train, trajs_test, trajs_val
    global feature_vals_train, feature_vals_test, feature_vals_val
    global greater_train_adjs, greater_test_adjs, greater_val_adjs
    global lesser_train_adjs, lesser_test_adjs, lesser_val_adjs
    
    # check
    if (split_train + split_test + split_val != size):
        print("ERROR: split_train + split_val + split_test != size")
        print("split_train: ", split_train)
        print("split_test", split_test)
        print("split_val:", split_val)
        print("size:", size)
        return
    else:
        print("traj sizes all good")
    if (split_lang_train + split_lang_test + split_lang_val != size_lang):
        print("ERROR: split_lang_train + split_lang_test + split_lang_val != size_lang")
        print("split_lang_train:", split_lang_train)
        print("split_lang_test", split_lang_test)
        print("split_lang_val:", split_lang_val)
        print("size lang: ", size_lang)
        return
    else:
        print("lang sizes all good")

    # random split_train number of integers from 0 to 323
    train_indices = np.random.choice(size, int(split_train), replace=False)
    val_test_indices = np.setdiff1d(np.arange(size), train_indices)
    test_indices = np.random.choice(val_test_indices, int(split_test), replace=False)
    val_indices = np.setdiff1d(val_test_indices, test_indices)

    if (len(train_indices) + len(val_indices) + len(test_indices) != int(size)):
        print("ERROR: split_train + split_val + split_test != size")

    print("train indices")
    print(train_indices)
    print("test_indices:")
    print(test_indices)
    print("val_indices")
    print(val_indices)

    # trajs
    trajs_train = [trajs[i] for i in train_indices]
    trajs_test = [trajs[i] for i in test_indices]
    trajs_val = [trajs[i] for i in val_indices]

    # feature_vals
    feature_vals_train = [feature_vals[i] for i in train_indices]
    feature_vals_test = [feature_vals[i] for i in test_indices]
    feature_vals_val = [feature_vals[i] for i in val_indices]

    # we don't use observations for now, so just save for training later
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(cur_dir, "../dataset")
    train_dir = os.path.join(dir, "train")
    test_dir = os.path.join(dir, "test")
    val_dir = os.path.join(dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir) or not os.path.exists(val_dir):
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

    # save feature_vals
    np.save(os.path.join(train_dir, "feature_vals.npy"), np.array(feature_vals_train))
    np.save(os.path.join(test_dir, "feature_vals.npy"), np.array(feature_vals_test))
    np.save(os.path.join(val_dir, "feature_vals.npy"), np.array(feature_vals_val))

    # observations (don't need since trajs = (obs + actions))
    observations_train = [observations[i] for i in train_indices]
    observations_test = [observations[i] for i in test_indices]
    observations_val = [observations[i] for i in val_indices]
    if (os.path.exists(dir) == False):
        os.makedirs(dir)
    if (os.path.exists(train_dir) == False):
        os.makedirs(train_dir)
    if (os.path.exists(test_dir) == False):
        os.makedirs(test_dir)
    if (os.path.exists(val_dir) == False):
        os.makedirs(val_dir)
    np.save(os.path.join(train_dir, "trajs.npy"), np.array(trajs_train))
    np.save(os.path.join(test_dir, "trajs.npy"), np.array(trajs_test))
    np.save(os.path.join(val_dir, "trajs.npy"), np.array(trajs_val))

    # actions
    actions_train = [actions[i] for i in train_indices]
    actions_test = [actions[i] for i in test_indices]
    actions_val = [actions[i] for i in val_indices]
    # we don't use actions for now, so just save for training later
    np.save(os.path.join(train_dir, "actions.npy"), np.array(actions_train))
    np.save(os.path.join(test_dir, "actions.npy"), np.array(actions_test))
    np.save(os.path.join(val_dir, "actions.npy"), np.array(actions_val))

    feature_vals = []

    # ------------------------------

    # now read in images.npy and save them as npy
    images = np.load(os.path.join(cur_dir, "../dataset/images.npy"))
    train_img_obs = np.array([images[i] for i in train_indices])
    test_img_obs = np.array([images[i] for i in test_indices])
    val_img_obs = np.array([images[i] for i in val_indices])
    print("train img shape")
    print(train_img_obs.shape)
    print("test img os shape")
    print(test_img_obs.shape)
    print(val_img_obs.shape)
    np.save(os.path.join(train_dir, "traj_img_obs.npy"), train_img_obs)
    np.save(os.path.join(test_dir, "traj_img_obs.npy"), test_img_obs)
    np.save(os.path.join(val_dir, "traj_img_obs.npy"), val_img_obs)

    # ------------------------------

    # get indices
    split_lang_train /= len(features)
    split_lang_test /= len(features)
    split_lang_val /= len(features)
    size_lang /= len(features)
    train_indices = np.random.choice(int(size_lang), int(split_lang_train), replace=False)
    val_test_indices = np.setdiff1d(np.arange(size_lang), train_indices)
    test_indices = np.random.choice(val_test_indices, int(split_lang_test), replace=False)
    val_indices = np.setdiff1d(val_test_indices, test_indices)

    train_indices = [int(i) for i in train_indices]
    test_indices = [int(i) for i in test_indices]
    val_indices = [int(i) for i in val_indices]

    if (len(train_indices) + len(val_indices) + len(test_indices) != size_lang):
        print("ERROR: len(train_indices) + len(val_indices) + len(test_indices) != size_lang")
        print("len train indices:" + len(train_indices))
        print("len test_indices: " + len(test_indices))
        print("len val indices:" + len(val_indices))

    print("train indices:")
    print(train_indices)
    print("test_indices:")
    print(test_indices)
    print("val_indices")
    print(val_indices)

    print("size of greater_adjs[0]")
    print(len(greater_adjs[0]))
    # greater_adjs
    # train
    greater_train_0 = [greater_adjs[0][int(i)] for i in train_indices]
    greater_train_1 = [greater_adjs[1][int(i)] for i in train_indices]
    greater_train_2 = [greater_adjs[2][int(i)] for i in train_indices]
    # greater_train_3 = [greater_adjs[3][int(i)] for i in train_indices]
    greater_train_adjs = [greater_train_0] + [greater_train_1] + [greater_train_2]
    # test
    greater_test_0 = [greater_adjs[0][int(i)] for i in test_indices]
    greater_test_1 = [greater_adjs[1][int(i)] for i in test_indices]
    greater_test_2 = [greater_adjs[2][int(i)] for i in test_indices]
    # greater_test_3 = [greater_adjs[3][int(i)] for i in test_indices]
    greater_test_adjs = [greater_test_0] + [greater_test_1] + [greater_test_2]
    # val
    greater_val_0 = [greater_adjs[0][int(i)] for i in val_indices]
    greater_val_1 = [greater_adjs[1][int(i)] for i in val_indices]
    greater_val_2 = [greater_adjs[2][int(i)] for i in val_indices]
    # greater_val_3 = [greater_adjs[3][int(i)] for i in val_indices]
    greater_val_adjs = [greater_val_0] + [greater_val_1] + [greater_val_2]

    greater_adjs = []

    # lesser_adjs
    # train
    lesser_train_0 = [lesser_adjs[0][int(i)] for i in train_indices]
    lesser_train_1 = [lesser_adjs[1][int(i)] for i in train_indices]
    lesser_train_2 = [lesser_adjs[2][int(i)] for i in train_indices]
    # lesser_train_3 = [lesser_adjs[3][int(i)] for i in train_indices]
    lesser_train_adjs = [lesser_train_0] + [lesser_train_1] + [lesser_train_2]
    # test
    lesser_test_0 = [lesser_adjs[0][int(i)] for i in test_indices]
    lesser_test_1 = [lesser_adjs[1][int(i)] for i in test_indices]
    lesser_test_2 = [lesser_adjs[2][int(i)] for i in test_indices]
    # lesser_test_3 = [lesser_adjs[3][int(i)] for i in test_indices]
    lesser_test_adjs = [lesser_test_0] + [lesser_test_1] + [lesser_test_2]
    # val
    lesser_val_0 = [lesser_adjs[0][int(i)] for i in val_indices]
    lesser_val_1 = [lesser_adjs[1][int(i)] for i in val_indices]
    lesser_val_2 = [lesser_adjs[2][int(i)] for i in val_indices]
    # lesser_val_3 = [lesser_adjs[3][int(i)] for i in val_indices]
    lesser_val_adjs = [lesser_val_0] + [lesser_val_1] + [lesser_val_2]

    lesser_adjs = []




# reads states and actions from files and returns them as a list of trajectories
    # every trajectory = (observations + actions) = (500, 43) for 500 timesteps
    # observations = (500, 39) for 500 timesteps
    # actions = (500, 4) for 500 timesteps
# also calculates feature values for each trajectory and returns them
def form_trajectories():
    print("\t-->>-- Forming Trajectories --<<--")
    
    formatted_trajs = []
    f_vals = []
    all_actions = []
    all_observations = []

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
                reward, avg_sum, tcp_height, tcp_vel, tcp_to_obj, env_state = env.compute_reward_v2(actions[step], obs)
                # env state (tuple) = joint state[0], mocap state (camera)[1]
                    # joint_state (23) = time(1) + pos(10) + vel(10) + action,udd(2)
                    # keep only pos and vel
                # import ipdb; ipdb.set_trace()
                joint_pos = env_state[0][1]
                joint_vel = env_state[0][2]

                # METHOD 2
                # obs = np.array(obs)
                cur_tcp_pos = obs[0:3]
                prev_tcp_pos = obs[18:21]
                pos_diff = cur_tcp_pos - prev_tcp_pos
                # 3
                obs = np.append(obs, pos_diff)



                # METHOD 1
                # obs = np.array(obs)
                # # original obs (39) = curr obs(18) + prev obs(18) + goal(3)
                #     # remove object quat and padding
                #     # remove prev obs
                #     # cur obs (18) = tcp pos(3) + grip dist(1) + obj pos(3) + obj quat(4) + obj padding(7)
                #     # prev obs (18)
                #     # goal (3)
                # obs = np.append(obs[0:7], obs[-3:39])
                # obs = np.concatenate([obs, np.array(joint_pos), np.array(joint_vel)])
                # # append velocity (1) to observation (39) = (40) dim
                # obs = np.append(obs, tcp_vel)
                observations.append(obs)
                
                avg_sum_vals.append(avg_sum)
                height_vals.append(tcp_height)
                velocity_vals.append(tcp_vel)
                distance_to_obj_vals.append(tcp_to_obj)
            
            # Add formatted trajectory
            observations = np.array(observations)
            actions = np.array(actions)

            traj = np.concatenate((observations, actions), axis=-1)
            formatted_trajs.append(traj)
            all_observations.append(observations)
            all_actions.append(actions)

            # Add feature values
            f_vals.append([height_vals] + [velocity_vals] + [distance_to_obj_vals] + [avg_sum_vals])

    # print("NUM_TIMESTEPS: " + str(NUM_TIMESTEPS))
    # print("fvals shape: ")
    # print(np.array(f_vals).shape)
    return formatted_trajs, f_vals, all_observations, all_actions

# modifies global variables greater_adjs and lesser_adjs with gpt augmented dataset json file
def get_gpt_dataset():
    global greater_adjs, lesser_adjs, all_adjs

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(cur_dir, "../dataset")

    with open(os.path.join(dir, "ver2_gpt_augmented_dataset_metaworld.json"), 'r') as openfile:
        gpt_dataset = json.load(openfile)
    
    greater_adjs = [gpt_dataset["greater_height"]] + [gpt_dataset["greater_velocity"]] + [gpt_dataset["greater_distance"]]
    lesser_adjs = [gpt_dataset["lesser_height"]] + [gpt_dataset["lesser_velocity"]] + [gpt_dataset["lesser_distance"]]
    all_adjs = greater_adjs[0] + greater_adjs[1] + greater_adjs[2] + lesser_adjs[0] + lesser_adjs[1] + lesser_adjs[2]

    all_adjs = list(sorted(set(all_adjs)))
    np.save(os.path.join(dir, "all_adjs.npy"), np.array(all_adjs))

    print("len greater: ", len(greater_adjs))
    print("len lesser:", len(lesser_adjs))

# i, j = indices of traj in order of ["0-0-0-0", "0-0-0-1", ... "2-2-2-2"], where each variant has 4 trials
# feature = {'height', 'velocity', 'distance_to_object', 'sum'}
# noisy = if True, 1% chance of opposite comparison
# returns string of language feedback of traj i --> traj j
def generate_synthetic_lang_feedback(i, j, feature, split, noisy=False):
    if (split == "train"):
        feature_vals = feature_vals_train
        greater_adjs = greater_train_adjs
        lesser_adjs = lesser_train_adjs
    elif (split == "test"):
        feature_vals = feature_vals_test
        greater_adjs = greater_test_adjs
        lesser_adjs = lesser_test_adjs
    elif (split == "val"):
        feature_vals = feature_vals_val
        greater_adjs = greater_val_adjs
        lesser_adjs = lesser_val_adjs

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

    # check what language feedback to give
    if (avg_i > avg_j):
        # i is greater
        if (noisy and np.random.rand() < prob):
            # by prob chance, return the opposite comparison\
            # j is greater
            temp = []
            for i in range(5):
                temp.append(greater_adjs[f][np.random.randint(len(greater_adjs[f]))])
            return temp
        else:
            # i is greater
            temp = []
            for i in range(5):
                temp.append(lesser_adjs[f][np.random.randint(len(lesser_adjs[f]))])
            return temp
    else:
        # j is greater
        if (noisy and np.random.rand() < prob):
            # by prob chance, return the opposite comparison
            # i is greater
            # print("some adj: " + str(lesser_adjs[f][np.random.randint(len(lesser_adjs[f]))]))
            temp = []
            for i in range(5):
                temp.append(lesser_adjs[f][np.random.randint(len(lesser_adjs[f]))])
            return temp
        else:
            # j is greater
            temp = []
            for i in range(5):
                temp.append(greater_adjs[f][np.random.randint(len(greater_adjs[f]))])
            return temp

def get_comparisons(i, j, split, noisy=False):
    out = []
    for feature in ['height', 'velocity', 'distance_to_object']:
    # for feature in ['velocity']:
        out.extend(generate_synthetic_lang_feedback(i, j, feature, split, noisy=noisy))
    return out


def initialize_globals(use_gpt_dataset, split_train, split_test, split_val, split_lang_train, split_lang_test, split_lang_val):
    global env, trajs, feature_vals, index_weights, observations, actions

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
    trajs, feature_vals, observations, actions = form_trajectories()

    # this modifies greater_adjs and lesser_adjs
    if (use_gpt_dataset):
        get_gpt_dataset()

    # split the datasets here for trajs, feature_vals, greater_adjs, lesser_adjs
    split_dataset(split_train=split_train, split_test=split_test, split_val=split_val, 
                  size=len(trajs), 
                  split_lang_train=split_lang_train, 
                  split_lang_test=split_lang_test, 
                  split_lang_val=split_lang_val, 
                  size_lang=len(greater_adjs[0] + greater_adjs[1]))


# generate traj a's, traj b's, and comparisons from a --> b
def generate_dataset(split, noisy=False, id_mapping=False, all_pairs=True):
    if (split == "train"):
        input_trajs = trajs_train
    elif (split == "test"):
        input_trajs = trajs_test
    elif (split == "val"):
        input_trajs = trajs_val

    # a --> b with language feedback comps
    dataset_traj_as = []
    dataset_traj_bs = []
    dataset_comps = []

    if (all_pairs):
        print("\t-->>-- Generating Dataset (all pairs)! --<<--")
        # all pairs where input_trajs = (0-0-0-0, ..., 2-2-2-2) each with 4 trials, as specified for run_traj_sbx.py
        for i in range(0, len(input_trajs)):
            for j in range(i + 1, len(input_trajs)):
                comps = get_comparisons(i, j, split, noisy=noisy)
                flipped_comps = get_comparisons(j, i, split, noisy=noisy)

                for index, c in enumerate(comps):
                    if id_mapping:
                        dataset_traj_as.append(i)
                        dataset_traj_bs.append(j)
                        dataset_comps.append(c)

                    else:
                        dataset_traj_as.append(input_trajs[i])
                        dataset_traj_bs.append(input_trajs[j])
                        dataset_comps.append(c)

                for index, fc in enumerate(flipped_comps):
                    if id_mapping:
                        dataset_traj_as.append(j)
                        dataset_traj_bs.append(i)
                        dataset_comps.append(fc)

                    else:
                        dataset_traj_as.append(input_trajs[j])
                        dataset_traj_bs.append(input_trajs[i])
                        dataset_comps.append(fc)

                    
    else:
        print("\t-->>-- Generating Dataset (random pairs)! --<<--")
        # random pairs
        for n in range(0, len(input_trajs)):
            i = 0
            j = 0
            while i == j:
                i = np.random.randint(len(input_trajs))
                j = np.random.randint(len(input_trajs))

            comps = get_comparisons(i, j, split, noisy=noisy)
            flipped_comps = get_comparisons(j, i, split, noisy=noisy)

            for index, c in enumerate(comps):
                if (id_mapping):
                    dataset_traj_as.append(i)
                    dataset_traj_bs.append(j)
                    dataset_comps.append(c)
                else:
                    dataset_traj_as.append(input_trajs[i])
                    dataset_traj_bs.append(input_trajs[j])
                    dataset_comps.append(c)

            for index, fc in enumerate(flipped_comps):
                if (id_mapping):
                    dataset_traj_as.append(j)
                    dataset_traj_bs.append(i)
                    dataset_comps.append(fc)

                else:
                    dataset_traj_as.append(input_trajs[j])
                    dataset_traj_bs.append(input_trajs[i])
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
    parser.add_argument('--split-train', type=float, default=260, help='number of trajectories for train set')
    parser.add_argument('--split-test', type=float, default=32, help='number of trajectories for test set')
    parser.add_argument('--split-val', type=float, default=32, help='number of trajectories for val set')
    parser.add_argument('--split-lang-train', type=float, default=304, help='number of language feedback phrases (only counting greater_adjs) for train set')
    parser.add_argument('--split-lang-test', type=float, default=48, help='number of language feedback phrases (only counting greater_adjs) for test set')
    parser.add_argument('--split-lang-val', type=float, default=48, help='number of language feedback phrases (only counting greater_adjs) for val set')


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
    use_gpt_dataset = args.use_gpt_dataset
    split_train = args.split_train
    split_test = args.split_test
    split_val = args.split_val
    split_lang_train = args.split_lang_train
    split_lang_test = args.split_lang_test
    split_lang_val = args.split_lang_val

    np.random.seed(seed)
    initialize_globals(use_gpt_dataset=use_gpt_dataset, split_train=split_train, split_test=split_test, split_val=split_val, split_lang_train=split_lang_train, split_lang_test=split_lang_test, split_lang_val=split_lang_val)

    
    dataset_train_traj_as, dataset_train_traj_bs, dataset_train_comps = generate_dataset(split="train", noisy=noise_augmentation, id_mapping=id_mapping, all_pairs=all_pairs)
    dataset_test_traj_as, dataset_test_traj_bs, dataset_test_comps = generate_dataset(split="test", noisy=noise_augmentation, id_mapping=id_mapping, all_pairs=all_pairs)
    dataset_val_traj_as, dataset_val_traj_bs, dataset_val_comps = generate_dataset(split="val", noisy=noise_augmentation, id_mapping=id_mapping, all_pairs=all_pairs)



    # dataset size:
        # 324 trajs (81 variants x 4 trials each)
            # 500 timesteps
        # 440 language feedback (55 sentences x 4 features x 2 (greater/lesser))
        # 269360 (len of dataset_train_traj_as) is from:
            # split:    
                # 324 trajs = 260 train + 32 test + 32 val
                    # random split
                # 440 lang = (172 train + 24 test + 24 val) x 2 (greater/lesser)
                    # train
                        # 43 greater height
                        # 43 greater velocity
                        # 43 greater distance
                        # 43 greater avg sum
                        # 43 lesser height
                        # 43 lesser velocity
                        # 43 lesser distance
                        # 43 lesser avg sum
                    # test
                        # 6 each
                    # val
                        # 6 each
            # train
                # 260 split trajs choose (i-->j) = 33670
                # 33670 x 4 features = 134680
                # 134680 x 2 for flipped order (j--i) = 269360
            # test
                # etc
            # val
                # etc
    # example dataset
        # dataset_train_traj_as.npy: 
            # contains indices or trajectories (observations + actions)
            # e.g., [0, 0, 0, 0, 1, ..., 259, 259, 259, 259]
        # dataset_train_traj_bs.npy: 
            # contains indices or trajectories (observations + actions)
            # e.g., [1, 1, 1, 1, 0, ..., 258, 258, 258, 258]
        # dataset_train_comps.npy: 
            # contains language feedback phrases to map trajectory a's --> trajectory b's for every feature ['height', 'velocity', 'distance_to_object', 'sum'], possibly with error if noisy
            # e.g., ["Move more down", "Move faster", "Move closer to the button", "Move better", ..., ...]
        # dataset_test_traj_as.npy
            # etc
        # dataset_test_traj_bs.npy
            # etc
        # dataset_test_comps.npy
            # etc
        # dataset_val_traj_as.npy
            # etc
        # dataset_val_traj_bs.npy
            # etc
        # dataset_val_comps.npy
            # etc

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(cur_dir, "../dataset")
    train_dir = os.path.join(dir, "train")
    test_dir = os.path.join(dir, "test")
    val_dir = os.path.join(dir, "val")
    if (os.path.exists(dir) == False):
            os.makedirs(dir)
            os.makedirs(train_dir)
            os.makedirs(test_dir)
            os.makedirs(val_dir)

    np.save(os.path.join(train_dir, "traj_a_indexes.npy"), np.array(dataset_train_traj_as))
    np.save(os.path.join(train_dir, "traj_b_indexes.npy"), np.array(dataset_train_traj_bs))
    # np.save(os.path.join(train_dir, "unique_nlcomps.npy"), np.array(dataset_train_comps))
    with open(os.path.join(train_dir, "nlcomps.json"), 'w') as openfile:
        json.dump(dataset_train_comps, openfile)
    # np.save(os.path.join(train_dir, "nlcomp_indexes.npy"), np.array(dataset_train_comp_indices))  
    # saved earlier in split_dataset()
    # np.save(os.path.join(train_dir, "trajs.npy"), np.array(observations_train))
    # np.save(os.path.join(train_dir, "actions.npy"), np.array(actions_train))

    np.save(os.path.join(test_dir, "traj_a_indexes.npy"), np.array(dataset_test_traj_as))
    np.save(os.path.join(test_dir, "traj_b_indexes.npy"), np.array(dataset_test_traj_bs))
    # np.save(os.path.join(test_dir, "unique_nlcomps.npy"), np.array(dataset_test_comps))
    with open(os.path.join(test_dir, "nlcomps.json"), 'w') as openfile:
        json.dump(dataset_test_comps, openfile)
    # np.save(os.path.join(test_dir, "nlcomp_indexes.npy"), np.array(dataset_test_comp_indices))
    # saved earlier in split_dataset()
    # np.save(os.path.join(test_dir, "trajs.npy"), np.array(observations_test))
    # np.save(os.path.join(test_dir, "actions.npy"), np.array(actions_test))

    np.save(os.path.join(val_dir, "traj_a_indexes.npy"), np.array(dataset_val_traj_as))
    np.save(os.path.join(val_dir, "traj_b_indexes.npy"), np.array(dataset_val_traj_bs))
    # np.save(os.path.join(val_dir, "unique_nlcomps.npy"), np.array(dataset_val_comps))
    with open(os.path.join(val_dir, "nlcomps.json"), 'w') as openfile:
        json.dump(dataset_val_comps, openfile)
    # np.save(os.path.join(val_dir, "nlcomp_indexes.npy"), np.array(dataset_val_comp_indices))
    # saved earlier in split_dataset()
    # np.save(os.path.join(val_dir, "trajs.npy"), np.array(observations_val))
    # np.save(os.path.join(val_dir, "actions.npy"), np.array(actions_val))

    print("-->>-->>-- Done making dataset!!! --<<--<<--")

    # ----------------------------

    # open np files and check
    print("-->>-->>-- Checking saved files... --<<--<<--")
    print("dataset_train_traj_as:")
    traj_as = np.load(os.path.join(train_dir, "traj_a_indexes.npy"))
    print(traj_as)
    print(traj_as.shape)
    # for i in range(0, 17):
    #     print(traj_as[i])

    print("dataset_train_traj_bs:")
    traj_bs = np.load(os.path.join(train_dir, "traj_b_indexes.npy"))
    print(traj_bs)
    print(traj_bs.shape)
    # for i in range(0, 17):
    #     print(traj_bs[i])

    print("dataset_train_comps:")
    with open(os.path.join(train_dir, "nlcomps.json"), 'r') as file:
        comps = json.load(file)
        # print(comps)
    # print(comps.shape)

    # -----------

    print("dataset_test_traj_as:")
    traj_as = np.load(os.path.join(test_dir, "traj_a_indexes.npy"))
    print(traj_as)
    print(traj_as.shape)
    # for i in range(0, 17):
    #     print(traj_as[i])

    print("dataset_test_traj_bs:")
    traj_bs = np.load(os.path.join(test_dir, "traj_b_indexes.npy"))
    print(traj_bs)
    print(traj_bs.shape)
    # for i in range(0, 17):
    #     print(traj_bs[i])

    print("dataset_test_comps:")
    with open(os.path.join(test_dir, "nlcomps.json"), 'r') as file:
        comps = json.load(file)
        # print(comps)
    # print(comps.shape)

    # -----------

    print("dataset_val_traj_as:")
    traj_as = np.load(os.path.join(val_dir, "traj_a_indexes.npy"))
    print(traj_as)
    print(traj_as.shape)
    # for i in range(0, 17):
    #     print(traj_as[i])

    print("dataset_val_traj_bs:")
    traj_bs = np.load(os.path.join(val_dir, "traj_b_indexes.npy"))
    print(traj_bs)
    print(traj_bs.shape)
    # for i in range(0, 17):
    #     print(traj_bs[i])

    print("dataset_val_comps:")
    with open(os.path.join(val_dir, "nlcomps.json"), 'r') as file:
        comps = json.load(file)
        # print(comps)
    # print(comps.shape)
    print("-->>-->>-- Check ^^^ everything looks okay?? --<<--<<--")