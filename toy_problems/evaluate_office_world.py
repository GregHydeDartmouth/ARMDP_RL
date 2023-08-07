import csv
import numpy as np
from collections import deque
from office_world import Actions
from office_world import OfficeWorld
from abstraction_machines.abstract_agent import AbstractAgent

rm = 3
actions = [a.value for a in Actions]
ow = OfficeWorld(rf_id=rm)
max_steps = 200

trials = 10
all_trial_rewards = []
all_trial_depths = []
all_trial_exemplars = []
all_trial_times = []
for t in range(0, trials):
    aa = AbstractAgent(actions, granularity='state', monotonic_levels=True, learning_rate=0.1, discount_factor=0.95)
    rewards = deque(maxlen=100)
    trial_rewards = deque(maxlen=100000)
    trial_depths = deque(maxlen=100000)
    trial_exemplars = deque(maxlen=100000)
    trial_times = deque(maxlen=100000)
    i = 0
    while i < 80000:
        aa.exploration_rate = 0.1
        # don't start counting i until we have a conflict. Serves as a sliding window. In resulting output, conflict will resolve SOMEWHERE in first 20,000 trajs.
        if aa.depth == 1 or len(trial_rewards) <= 20000:
            i = 0
            if rm == 1:
                aa.exploration_rate = 0.2
        state = ow.reset()
        aa.reset()
        ep_reward = 0
        if i > 70000:
            aa.exploration_rate = 0
        for j in range(0, max_steps):
            action = aa.choose_action(state)
            reward, next_state, done = ow.execute_action(action)
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
    all_trial_rewards.append(list(trial_rewards))
    all_trial_depths.append(list(trial_depths))
    all_trial_exemplars.append(list(trial_exemplars))
    all_trial_times.append(list(trial_times))

with open('data/rm_{}_abstraction_reward_averages.csv'.format(rm), 'w', newline='') as f:
    writer = csv.writer(f)
    for trial_rewards in all_trial_rewards:
        writer.writerow(trial_rewards)
with open('data/rm_{}_abstraction_depths.csv'.format(rm), 'w', newline='') as f:
    writer = csv.writer(f)
    for trial_depths in all_trial_depths:
        writer.writerow(trial_depths)
with open('data/rm_{}_abstraction_exemplars.csv'.format(rm), 'w', newline='') as f:
    writer = csv.writer(f)
    for trial_exemplars in all_trial_exemplars:
        writer.writerow(trial_exemplars)
with open('data/rm_{}_abstraction_solve_times.csv'.format(rm), 'w', newline='') as f:
    writer = csv.writer(f)
    for trial_times in all_trial_times:
        writer.writerow(trial_times)
#aa.graph_AMDP()
#aa.graph_RM()
#aa.save_triggers('rm_{}.json'.format(rm))
