import json
from abstraction_machines.abstract_machine import AbstractionMachine
from beachworld import Beachworld
from collections import deque

run_is = []
run_depths = []
run_reward_avgs = []
run_solve_times = []
for run in range(0, 10):
    _is = []
    depths = []
    reward_avgs = []
    solve_times = []
    BW = Beachworld()
    gamma = 0.9
    eps = 0.3
    action_set=['^','v','<','>','o']
    AM = AbstractionMachine(action_set=action_set, granularity='triple')
    rewards = deque(maxlen=100)

    for i in range(0, 10000):
        state = BW.reset()
        AM.reset()
        done = False

        trajectory_reward = 0
        if i > 9000:
            eps = 0
        while not done:
            action = AM.get_action(state, eps=eps)
            reward, next_state, done = BW.step(action)
            trajectory_reward += reward
            AM.step(state, action, reward, next_state)
            state = next_state
        rewards.append(trajectory_reward)
        avg_reward = sum(rewards)/len(rewards)
        _is.append(i)
        depths.append(AM.depth)
        reward_avgs.append(avg_reward)
        t = 0
        for depth, time in AM.depth_solve_time.items():
            t += time
        solve_times.append(t)
        print('\ravg reward: {}, episode_num: {}'.format(avg_reward, i), end="")
    run_is.append(_is)
    run_depths.append(depths)
    run_reward_avgs.append(reward_avgs)
    run_solve_times.append(solve_times)
x = {'is':run_is,
     'depths':run_depths,
     'run_reward_avgs':run_reward_avgs,
     'run_solve_times':run_solve_times}
with open('beachworld.json', 'w') as f:
    json.dump(x, f)
