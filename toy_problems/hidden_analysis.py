import json
import numpy as np
from dqn.dqn import DQN
from collections import deque
from office_world import Actions
from office_world import OfficeWorld

granularity = 'state'
actions = [a.value for a in Actions]
ow = OfficeWorld(rf_id=1)
with open('toy_problems/models/rm_1.json', 'r') as f:
    triggers = json.load(f)
trigger_state = dict()
for trigger, level in triggers.items():
    trigger_state[trigger] = 0
state_space = len(triggers.keys()) + 2
dqn_agent = DQN(state_space, len(actions))

rewards = deque(maxlen=100)
max_steps = 200

for i in range(0, 100000):
    for trigger, state in trigger_state.items():
        state = 0
    state = list(ow.reset())
    ab_state = np.array(list(state) + list(trigger_state.values()))
    ep_reward = 0
    level = 0
    for j in range(0, max_steps):
        action, _ = dqn_agent.act(ab_state)
        reward, next_state, done = ow.execute_action(action)
        if reward > 0.5:
            print(ab_state)
        if granularity == 'state':
            check_trigger = '{},{}'.format(level, next_state)
        elif granularity == 'triple':
            check_trigger = '{}^{},{},{}'.format(state, level, action, next_state)
        if check_trigger in triggers:
            print(check_trigger)
            x = input()
            level = triggers[check_trigger]
            trigger_state[check_trigger] = 1
        ab_next_state = np.array(list(next_state) + list(trigger_state.values()))
        dqn_agent.step(ab_state, np.array([action]), np.array([reward]), ab_next_state, np.array([done]))
        state = next_state
        ab_state = ab_next_state
        ep_reward += reward
        if done:
            break
    rewards.append(ep_reward)
    reward_avg = np.mean(rewards)
    print("\rEpisode: {} Average reward: {}".format(i, reward_avg), end='')
