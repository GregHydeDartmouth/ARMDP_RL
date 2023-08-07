import os
import ast
import csv
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import json
import torch
import random
import numpy as np
from util import encode_state
from dqn.lstmdqn import LSTMDQN
from collections import deque
from office_world import Actions
from office_world import OfficeWorld

rm = 4
actions = [a.value for a in Actions]
ow = OfficeWorld(rf_id=rm)
state_space = 12*9

with open('data/rm_{}_optimal_trajectory.tsv'.format(rm), 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    optimal_trajectory = []
    for row in reader:
        state, action, reward, next_state, done = row
        optimal_trajectory.append([ast.literal_eval(state), int(action), int(reward), ast.literal_eval(next_state), ast.literal_eval(done)])

sample_optimal_rate = 10

max_steps = 200
trials = 10

all_trials = []
for t in range(0, trials):
    seed = random.randint(0, 1000)
    print("\nSEED: {}".format(seed))
    dqn_agent = LSTMDQN(state_space, len(actions), lr=1e-3, tau=0.01, seed=seed)
    trial_averages = []
    rewards = deque(maxlen=1000)
    for i in range(0, 50000):
        # sample optimal
        if i % sample_optimal_rate == 0:
            for triple in optimal_trajectory:
                state, action, reward, next_state, done = triple
                dqn_agent.step(encode_state(state), np.array([action]), np.array([reward]), encode_state(next_state), np.array([done]))
            dqn_agent.reset()
        # normal run
        state = encode_state(ow.reset())
        h, c = dqn_agent.get_init_hidden_state()
        ep_reward = 0
        for j in range(0, max_steps):
            action, h, c = dqn_agent.act(state, h, c)
            reward, next_state, done = ow.execute_action(action)
            next_state = encode_state(next_state)
            dqn_agent.step(state, np.array([action]), np.array([reward]), next_state, np.array([done]))
            state = next_state
            ep_reward += reward
            if done:
                break
        dqn_agent.train()
        dqn_agent.reset()
        rewards.append(ep_reward)
        reward_avg = np.mean(rewards)
        print("\rEpisode: {} Average reward: {}".format(i, reward_avg), end='')
        trial_averages.append(reward_avg)
    all_trials.append(trial_averages)
with open('data/rm_{}_lstmdqn_reward_averages.csv'.format(rm), 'w', newline='') as f:
    writer = csv.writer(f)
    for trial_averages in all_trials:
        writer.writerow(trial_averages)
