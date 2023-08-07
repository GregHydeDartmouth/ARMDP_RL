import os
import csv
import ast
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import json
import torch
import random
import numpy as np
from util import encode_state
from dqn.dqn import DQN
from collections import deque
from office_world import Actions
from office_world import OfficeWorld

rm = 4
granularity = 'state'
actions = [a.value for a in Actions]
ow = OfficeWorld(rf_id=rm)
with open('models/rm_{}.json'.format(rm), 'r') as f:
    triggers = json.load(f)
trigger_state = dict()
for trigger, level in triggers.items():
    trigger_state[trigger] = 0
state_space = len(triggers.keys()) + 12*9

with open('data/rm_{}_optimal_trajectory.tsv'.format(rm), 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    optimal_trajectory = []
    for row in reader:
        state, action, reward, next_state, done = row
        optimal_trajectory.append([ast.literal_eval(state), int(action), int(reward), ast.literal_eval(next_state), ast.literal_eval(done)])


max_steps = 200
sample_optimal_rate = 10

trials = 10

all_trials = []
for t in range(0, trials):
    seed = random.randint(0, 1000)
    print("\nSEED: {}".format(seed))
    dqn_agent = DQN(state_space, len(actions), lr=1e-3, tau=0.01, seed=seed)
    trial_averages = []
    rewards = deque(maxlen=1000)
    for i in range(0, 50000):
        # sample optimal
        if i % sample_optimal_rate == 0:
            for trigger in trigger_state:
                trigger_state[trigger] = 0
            level = 0
            for triple in optimal_trajectory:
                state, action, reward, next_state, done = triple
                ab_state = np.append(encode_state(state), list(trigger_state.values()))
                if granularity == 'state':
                    check_trigger = '{},{}'.format(level, next_state)
                elif granularity == 'triple':
                    check_trigger = '{}^{},{},{}'.format(state, level, action, next_state)
                if check_trigger in triggers:
                    level = triggers[check_trigger]
                    trigger_state[check_trigger] = 1
                ab_next_state = np.append(encode_state(next_state), list(trigger_state.values()))
                dqn_agent.step(ab_state, np.array([action]), np.array([reward]), ab_next_state, np.array([done]))
        # normal run
        for trigger in trigger_state:
            trigger_state[trigger] = 0
        state = ow.reset()
        ab_state = np.append(encode_state(state), list(trigger_state.values()))
        ep_reward = 0
        level = 0
        for j in range(0, max_steps):
            action = dqn_agent.act(ab_state)
            reward, next_state, done = ow.execute_action(action)
            if granularity == 'state':
                check_trigger = '{},{}'.format(level, next_state)
            elif granularity == 'triple':
                check_trigger = '{}^{},{},{}'.format(state, level, action, next_state)
            if check_trigger in triggers:
                level = triggers[check_trigger]
                trigger_state[check_trigger] = 1
            ab_next_state = np.append(encode_state(next_state), list(trigger_state.values()))
            dqn_agent.step(ab_state, np.array([action]), np.array([reward]), ab_next_state, np.array([done]))
            state = next_state
            ab_state = ab_next_state
            ep_reward += reward
            if done:
                break
            dqn_agent.train()
        rewards.append(ep_reward)
        reward_avg = np.mean(rewards)
        print("\rEpisode: {} Average reward: {}".format(i, reward_avg), end='')
        trial_averages.append(reward_avg)
    all_trials.append(trial_averages)
with open('data/rm_{}_dqn_with_ab_reward_averages.csv'.format(rm), 'w', newline='') as f:
    writer = csv.writer(f)
    for trial_averages in all_trials:
        writer.writerow(trial_averages)
