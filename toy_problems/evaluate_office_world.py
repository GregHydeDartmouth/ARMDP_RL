import numpy as np
from collections import deque
from office_world import Actions
from office_world import OfficeWorld
from abstraction_machines.abstract_agent import AbstractAgent

actions = [a.value for a in Actions]
ow = OfficeWorld(rf_id=4)
aa = AbstractAgent(actions, granularity='state', monotonic_levels=True)

rewards = deque(maxlen=100)

for i in range(0, 100000):
    state = ow.reset()
    aa.reset(make_graph_on_update=True)
    ep_reward = 0
    while True:
        action = aa.choose_action(state)
        reward, next_state, done = ow.execute_action(action)
        aa.step(state, action, reward, next_state)
        state = next_state
        ep_reward += reward
        if done:
            break
    rewards.append(ep_reward)
    reward_avg = np.mean(rewards)
    if reward_avg > 0.1:
        aa.exploration_rate = 0
    print("\rEpisode: {} Average reward: {}".format(i, reward_avg), end='')
aa.graph_AMDP()

