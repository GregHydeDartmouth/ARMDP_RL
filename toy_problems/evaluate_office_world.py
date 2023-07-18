from office_world import Actions
from office_world import OfficeWorld
from abstraction_machines.abstraction_agent import AbstractAgent

actions = [a.value for a in Actions]
ow = OfficeWorld(rf_id=2)
aa = AbstractAgent(actions)
for i in range(0, 1000000000):
    state = ow.reset()
    aa.reset()
    rewards = 0
    while True:
        action = aa.choose_action(state)
        reward, next_state, done = ow.execute_action(action)
        aa.step(state, action, reward, next_state)
        state = next_state
        rewards += reward
        if done:
            break

