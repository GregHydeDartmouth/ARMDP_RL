import os
import csv
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
from office_world import Actions
from office_world import OfficeWorld


rm = 4
actions = [a.value for a in Actions]
ow = OfficeWorld(rf_id=rm)

trajectory = []

state = list(ow.reset())
while True:
    ow.show()
    action = input('Enter action 0) up, 1) right, 2) down, 3) left, 4, stay: ')
    reward, next_state, done = ow.execute_action(int(action))
    trajectory.append([state, action, reward, next_state, done])
    state = next_state
    if done:
        break
with open('trajectory.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(trajectory)


