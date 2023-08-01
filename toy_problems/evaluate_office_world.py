import numpy as np
from collections import deque
from office_world import Actions
from office_world import OfficeWorld
from abstraction_machines.abstract_agent import AbstractAgent

actions = [a.value for a in Actions]
ow = OfficeWorld(rf_id=1)
aa = AbstractAgent(actions, granularity='state', monotonic_levels=True, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.2)
rewards = deque(maxlen=100)
max_steps = 200

for i in range(0, 100000):
    state = ow.reset()
    aa.reset(make_graph_on_update=True)
    ep_reward = 0
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
aa.graph_AMDP()
aa.graph_RM()

