import os
import csv
import numpy as np
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
import matplotlib.pyplot as plt

def get_files_for_rm(rm):
    file_types = ['dqn_with_ab', 'rdqn', 'lstmdqn', 'grudqn', 'dqn_no_ab']
    dqn_w_ab = []
    dqn_no_ab = []
    rdqn = []
    grudqn = []
    lstmdqn = []
    for file_type in file_types:
        with open('data/rm_{}_{}_reward_averages.csv'.format(rm, file_type), 'r') as f:
            reader = csv.reader(f)
            if file_type == 'dqn_with_ab':
                for row in reader:
                    dqn_w_ab.append(row)
            elif file_type == 'dqn_no_ab':
                for row in reader:
                    dqn_no_ab.append(row)
            elif file_type == 'rdqn':
                for row in reader:
                    rdqn.append(row)
            elif file_type == 'grudqn':
                for row in reader:
                    grudqn.append(row)
            elif file_type == 'lstmdqn':
                for row in reader:
                    lstmdqn.append(row)
    return np.array(dqn_w_ab, dtype=float), np.array(dqn_no_ab, dtype=float), np.array(rdqn, dtype=float), np.array(grudqn, dtype=float), np.array(lstmdqn, dtype=float)


rm_1_rewards = get_files_for_rm(1)
rm_2_rewards = get_files_for_rm(2)
rm_3_rewards = get_files_for_rm(3)
rm_4_rewards = get_files_for_rm(4)
rewards = [rm_1_rewards, rm_2_rewards, rm_3_rewards, rm_4_rewards]

fig, axes = plt.subplots(4, 1, figsize=(10, 15))
for i, r in enumerate(rewards):
    dqn_abs_i, dqn_no_abs_i, rdqn_i, grudqn_i, lstmdqn_i = r

    # episodes
    episodes = np.arange(dqn_abs_i.shape[1])

    # dqn w/ abs
    dqn_abs_i_means = np.mean(dqn_abs_i, axis=0)
    dqn_abs_i_stds = np.std(dqn_abs_i, axis=0)
    axes[i].plot(episodes, dqn_abs_i_means, label='dqn_w_abs')
    axes[i].fill_between(episodes, dqn_abs_i_means-dqn_abs_i_stds, dqn_abs_i_means+dqn_abs_i_stds, alpha=0.2)

    # dqn w/o abs
    dqn_no_abs_i_means = np.mean(dqn_no_abs_i, axis=0)
    dqn_no_abs_i_stds = np.std(dqn_no_abs_i, axis=0)
    axes[i].plot(episodes, dqn_no_abs_i_means, label='dqn_w/o_abs', marker = 'o', markevery = 2500, markersize = 8)
    axes[i].fill_between(episodes, dqn_no_abs_i_means-dqn_no_abs_i_stds, dqn_no_abs_i_means+dqn_no_abs_i_stds, alpha=0.2)

    # rdqn
    rdqn_i_means = np.mean(rdqn_i, axis=0)
    rdqn_i_stds = np.std(rdqn_i, axis=0)
    axes[i].plot(episodes, rdqn_i_means, label='rdqn', marker = '^', markevery = 2500, markersize = 8)
    axes[i].fill_between(episodes, rdqn_i_means-rdqn_i_stds, rdqn_i_means+rdqn_i_stds, alpha=0.2)

    # grudqn
    grudqn_i_means = np.mean(grudqn_i, axis=0)
    grudqn_i_stds = np.std(grudqn_i, axis=0)
    axes[i].plot(episodes, grudqn_i_means, label='grudqn', marker = 's', markevery = 2500, markersize = 8)
    axes[i].fill_between(episodes, grudqn_i_means-grudqn_i_stds, grudqn_i_means+grudqn_i_stds, alpha=0.2)

    # lstmdqn
    lstmdqn_i_means = np.mean(lstmdqn_i, axis=0)
    lstmdqn_i_stds = np.std(lstmdqn_i, axis=0)
    axes[i].plot(episodes, lstmdqn_i_means, label='lstmdqn', marker = 'D', markevery = 2500, markersize = 8)
    axes[i].fill_between(episodes, lstmdqn_i_means-lstmdqn_i_stds, lstmdqn_i_means+lstmdqn_i_stds, alpha=0.2)

axes[0].set_ylabel('Task b)')
axes[1].set_ylabel('Task c)')
axes[2].set_ylabel('Task d)')
axes[3].set_ylabel('Task e)')
axes[3].set_xlabel('Episodes')

lines, labels = axes[0].get_legend_handles_labels()
fig.legend(lines, labels, loc='upper right')

plt.savefig('figures/exp2_figure.png')
