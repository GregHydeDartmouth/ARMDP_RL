import numpy as np
from collections import deque
from office_world import Actions
from office_world import OfficeWorld
from abstraction_machines.abstract_agent import AbstractAgent

actions = [a.value for a in Actions]
ow = OfficeWorld(rf_id=1)
aa = AbstractAgent(actions)

rewards = deque(maxlen=100)

for i in range(0, 1000000000):
    state = ow.reset()
    aa.reset()
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
    print("\rEpisode: {} Average reward: {}".format(i, reward_avg), end='')

