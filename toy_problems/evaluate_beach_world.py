import json
from abstraction_machines.abstract_agent import AbstractAgent
from beachworld import Beachworld
from collections import deque

action_set=['^','v','<','>','o']
aa = AbstractAgent(action_set, granularity='triple', monotonic_levels=True, learning_rate=0.1, discount_factor = 0.95, exploration_rate = 0.4)
BW = Beachworld()
rewards = deque(maxlen=100)

for i in range(0, 10000):
    state = BW.reset()
    aa.reset()
    done = False
    trajectory_reward = 0
    if i > 9000:
        eps = 0
    while not done:
        action = aa.choose_action(state)
        reward, next_state, done = BW.step(action)
        trajectory_reward += reward
        aa.step(state, action, reward, next_state, done)
        state = next_state
    rewards.append(trajectory_reward)
    avg_reward = sum(rewards)/len(rewards)
    print('\ravg reward: {}, episode_num: {}'.format(avg_reward, i), end="")
aa.graph_AMDP()
aa.graph_RM()
