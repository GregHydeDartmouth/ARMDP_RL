import os
import csv
from breakfast_world import Actions, BreakfastWorld

rm = 4
actions = [a.value for a in Actions]
bw = BreakfastWorld(cumulative_reward = True)

trajectory = []
actions_dict = {'w' : 0,
                'a' : 3,
                's' : 2,
                'd' : 1,
                'x' : 4}

while True:
    state = bw.reset()
    cumulative_reward = 0
    while True:
        bw.show()
        action = input('Enter action \n w) up, \n d) right, \n s) down, \n a) left, \n x), stay: ')
        reward, next_state, done = bw.execute_action(actions_dict[action])
        cumulative_reward += reward
        print(reward, done)
        trajectory.append([state, action, reward, next_state, done])
        state = next_state
        if done:
            break


