import os
import csv
import signal
import numpy as np
from collections import deque
from breakfast_world import Actions, BreakfastWorld
from abstraction_machines.abstract_agent import AbstractAgent

actions = [a.value for a in Actions]
cumulative_only = True
rm = 2
bw = BreakfastWorld(cumulative_only=cumulative_only, rm=rm)
max_steps = 200
silent = True

trials = 10
all_trial_rewards = []
all_trial_depths = []
all_trial_exemplars = []
all_trial_times = []
for t in range(0, trials):
    aa = AbstractAgent(actions, granularity='state', monotonic_levels=True, learning_rate=0.1, discount_factor=0.95)
    rewards = deque(maxlen=100)
    trial_rewards = deque(maxlen=10000)
    trial_depths = deque(maxlen=10000)
    trial_exemplars = deque(maxlen=10000)
    trial_times = deque(maxlen=10000)
    i = 0
    while i < 8000:
        # don't start counting i until we have a conflict. Serves as a sliding window.
        if aa.depth == 1 or len(trial_rewards) <= 2000:
            i = 0
        state = bw.reset()
        aa.reset(silent=silent)
        ep_reward = 0
        if i > 7000:
            aa.exploration_rate = 0
        for j in range(0, max_steps):
            action = aa.choose_action(state)
            reward, next_state, done = bw.execute_action(action)
            aa.step(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        reward_avg = np.mean(rewards)
        print("\rEpisode: {} Average reward: {}".format(i, reward_avg), end='')
        trial_rewards.append(np.mean(rewards))
        trial_depths.append(aa.depth)
        trial_exemplars.append(len(aa.conflicting_trajectories))
        trial_times.append(aa.solve_time)
        i += 1
    #aa.graph_AMDP()
    #aa.graph_RM()
    all_trial_rewards.append(list(trial_rewards))
    all_trial_depths.append(list(trial_depths))
    all_trial_exemplars.append(list(trial_exemplars))
    all_trial_times.append(list(trial_times))

with open('data/bfw_rm_{}_cumulative_only_{}_abstraction_reward_averages.csv'.format(rm, cumulative_only), 'w', newline='') as f:
    writer = csv.writer(f)
    for trial_rewards in all_trial_rewards:
        writer.writerow(trial_rewards)
with open('data/bfw_rm_{}_cumulative_only_{}_abstraction_depths.csv'.format(rm, cumulative_only), 'w', newline='') as f:
    writer = csv.writer(f)
    for trial_depths in all_trial_depths:
        writer.writerow(trial_depths)
with open('data/bfw_rm_{}_cumulative_only_{}_abstraction_exemplars.csv'.format(rm, cumulative_only), 'w', newline='') as f:
    writer = csv.writer(f)
    for trial_exemplars in all_trial_exemplars:
        writer.writerow(trial_exemplars)
with open('data/bfw_rm_{}_cumulative_only_{}_abstraction_solve_times.csv'.format(rm, cumulative_only), 'w', newline='') as f:
    writer = csv.writer(f)
    for trial_times in all_trial_times:
        writer.writerow(trial_times)

'''
aa = AbstractAgent(actions, granularity='state', monotonic_levels=True, learning_rate=0.1, discount_factor=0.95)
for i in range(0, 10000):
    state = bw.reset()
    aa.reset(make_graph_on_update=True)
    for j in range(0, max_steps):
        action = aa.choose_action(state)
        reward, next_state, done = bw.execute_action(action)
        aa.step(state, action, reward, next_state, done)
        if done:
            break
        state = next_state
    print(i)
aa.graph_AMDP()
aa.graph_RM()
'''
