import csv
from abstraction_machines.abstract_machine import AbstractionMachine

trajectories = []
for i in range(1,5):
    with open('sample_trajectories/coffee_trajectories/t{}.csv'.format(i), 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        trajectory = []
        for row in csv_reader:
            trajectory.append(row)
        trajectories.append(trajectory)
AM = AbstractionMachine(trajectories)
AM.resolve_reward_conflicts(make_graph=True)
