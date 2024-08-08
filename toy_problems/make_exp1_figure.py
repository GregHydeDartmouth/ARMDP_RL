import os
import csv
import numpy as np
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
import matplotlib.pyplot as plt

def get_files_for_rm(rm):
    rewards = []
    depths = []
    exemplars = []
    solve_times = []
    file_types = ['reward_averages', 'depths', 'exemplars', 'solve_times']
    for file_type in file_types:
        with open('data/rm_{}_abstraction_{}.csv'.format(rm, file_type), 'r') as f:
            reader = csv.reader(f)
            if file_type == 'reward_averages':
                for row in reader:
                    rewards.append(row)
            elif file_type == 'depths':
                for row in reader:
                    depths.append(row)
            elif file_type == 'solve_times':
                for row in reader:
                    solve_times.append(row)
            elif file_type == 'exemplars':
                for row in reader:
                    exemplars.append(row)
    return np.array(rewards, dtype=float), np.array(depths, dtype=float), np.array(exemplars, dtype=float), np.array(solve_times, dtype=float)

# get data
rm_1_rewards, rm_1_depths, rm_1_exemplars, rm_1_solve_times = get_files_for_rm(1)
rm_2_rewards, rm_2_depths, rm_2_exemplars, rm_2_solve_times = get_files_for_rm(2)
rm_3_rewards, rm_3_depths, rm_3_exemplars, rm_3_solve_times = get_files_for_rm(3)
rm_4_rewards, rm_4_depths, rm_4_exemplars, rm_4_solve_times = get_files_for_rm(4)
rewards = [rm_1_rewards, rm_2_rewards, rm_3_rewards, rm_4_rewards]
depths = [rm_1_depths, rm_2_depths, rm_3_depths, rm_4_depths]
exemplars = [rm_1_exemplars, rm_2_exemplars, rm_3_exemplars, rm_4_exemplars]
solve_times = [rm_1_solve_times, rm_2_solve_times, rm_3_solve_times, rm_4_solve_times]
data = zip(rewards, depths, exemplars, solve_times)

fig, axes = plt.subplots(4, 4, sharex=False, sharey=False, figsize=(20, 10))

for i, d in enumerate(data):
    rewards_i, depths_i, exemplars_i, solve_times_i = d

    # episodes
    episodes = np.arange(rewards_i.shape[1])

    # plot rewards
    reward_means = np.mean(rewards_i, axis=0)
    reward_devs = np.std(rewards_i, axis=0)
    axes[i,0].plot(episodes, reward_means)
    axes[i,0].fill_between(episodes, reward_means-reward_devs, reward_means+reward_devs, alpha=0.2)
    axes[i,0].axhline(y=1, color='red', linewidth=1.5, label='optimal')
    axes[i,0].axhline(y=0, color='black', linestyle='-', linewidth=1)

    # plot depths
    depths_maxs = np.max(depths_i, axis=0)
    depths_mins = np.min(depths_i, axis=0)
    depths_means = np.mean(depths_i, axis=0)
    axes[i,1].plot(episodes, depths_maxs, label='max', linestyle = 'dashed')
    axes[i,1].plot(episodes, depths_mins, label='min', linestyle = 'dashdot')
    axes[i,1].plot(episodes, depths_means, label='avg', linestyle='dotted')

    # plot exemplars
    exemplar_maxs = np.max(exemplars_i, axis=0)
    exemplar_mins = np.min(exemplars_i, axis=0)
    exemplar_means = np.mean(exemplars_i, axis=0)
    axes[i,2].plot(episodes, exemplar_maxs, label='max', linestyle = 'dashed')
    axes[i,2].plot(episodes, exemplar_mins, label='min', linestyle = 'dashdot')
    axes[i,2].plot(episodes, exemplar_means, label='avg', linestyle='dotted')

    # plot solve_times
    solve_times_maxs = np.max(solve_times_i, axis=0)
    solve_times_mins = np.min(solve_times_i, axis=0)
    solve_times_means = np.mean(solve_times_i, axis=0)
    axes[i,3].plot(episodes, solve_times_maxs, label='max', linestyle = 'dashed')
    axes[i,3].plot(episodes, solve_times_mins, label='min', linestyle = 'dashdot')
    axes[i,3].plot(episodes, solve_times_means, label='avg', linestyle='dotted')

axes[0, 0].set_ylabel('Task b)')
axes[1, 0].set_ylabel('Task c)')
axes[2, 0].set_ylabel('Task d)')
axes[3, 0].set_ylabel('Task e)')
axes[-1, 0].set_xlabel('Episodes')
axes[-1, 1].set_xlabel('Episodes')
axes[-1, 2].set_xlabel('Episodes')
axes[-1, 3].set_xlabel('Episodes')
axes[0, 0].set_title('Rewards')
axes[0, 1].set_title('Depths')
axes[0, 2].set_title('Exemplars')
axes[0, 3].set_title('Cumulative Solve Time')

lines, labels = axes[1,1].get_legend_handles_labels()
fig.legend(lines, labels, loc='upper right')

plt.savefig('figures/exp1_figure.png')
