import numpy as np
import os
import json

cur_dir = os.path.dirname(os.path.abspath(__file__))
dir = os.path.join(cur_dir, "../dataset")
train_dir = os.path.join(dir, "train")

trajs = np.load(os.path.join(train_dir, "trajs.npy"))
trajs_a_indexes = np.load(os.path.join(train_dir, "traj_a_indexes.npy"))
trajs_b_indexes = np.load(os.path.join(train_dir, "traj_b_indexes.npy"))
f_vals = np.load(os.path.join(train_dir, "feature_vals.npy"))
nlcomp_indexes = np.load(os.path.join(train_dir, "nlcomp_indexes.npy"))
with open(os.path.join(train_dir, "unique_nlcomps.json"), 'r') as openfile:
    unique_nlcomps = json.load(openfile)

print("lenght:")
print(len(trajs_a_indexes))

# check first 5 comp pairs
for i in range(15):
    # print("Traj A Index: ", trajs_a_indexes[i])
    # print("Traj B Index: ", trajs_b_indexes[i])
    a_mean = np.mean(f_vals[trajs_a_indexes[len(trajs_a_indexes) - i - 1]][1])
    b_mean = np.mean(f_vals[trajs_b_indexes[len(trajs_b_indexes) - i - 1]][1])
    fval_a_mean = np.mean(trajs[trajs_a_indexes[len(trajs_a_indexes) - i - 1]][:, -5])
    fval_b_mean = np.mean(trajs[trajs_b_indexes[len(trajs_b_indexes) - i - 1]][:, -5])
    print("Feature Values a: ", a_mean)
    print("traj a velo fval:", fval_a_mean)
    print("Feature Values b: ", b_mean)
    print("traj b velo fval:", fval_b_mean)
    # print("NLcomp Indexes: ", nlcomp_indexes[i])
    print("NLcomp: ", unique_nlcomps[len(unique_nlcomps) - i - 1])
    print("-----------------------")
    # print("Trajectory a: ", trajs[trajs_a_indexes[i]])
    # print("Trajectory b: ", trajs[trajs_b_indexes[i]])