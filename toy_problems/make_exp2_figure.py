import os
import csv
import numpy as np
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
import matplotlib.pyplot as plt

model_types = ['dqn_w_ab', 'rdqn', 'lstmdqn', 'grudqn'] #, 'dqn_no_ab']

rm_4_models_rewards = []
for model_type in model_types:
    trial_rewards = []
    with open('data/rm_4_{}_reward_averages.csv'.format(model_type), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            trial_rewards.append(row)
    rm_4_models_rewards.append(trial_rewards)

for rm_4_model_rewards, model_name in zip(rm_4_models_rewards, model_types):
    rm_4_model_rewards = np.array(rm_4_model_rewards, dtype=float)
    column_means = np.mean(rm_4_model_rewards, axis=0)
    column_stddev = np.std(rm_4_model_rewards, axis=0)
    i = np.arange(rm_4_model_rewards.shape[1])
    plt.plot(i, column_means, label=model_name)
    plt.fill_between(i, column_means - column_stddev, column_means + column_stddev, alpha=0.2)
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.legend()
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.savefig('figures/test.png')
