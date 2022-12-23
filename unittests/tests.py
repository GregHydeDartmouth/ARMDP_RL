import unittest
from abstraction_machines.abstract_machine import AbstractionMachine

class test_abstract_q_learning(unittest.TestCase):
    def est_powerpoint_problem(self):
        actions = {'^' : '0',
                   'v' : '1',
                   '<' : '2',
                   '>' : '3'}

        trajectories = []
        t = [['1', actions['>'], 0, '2'],
              ['2', actions['^'], 0, '6'],
              ['6', actions['^'], 0, '10'],
              ['10', actions['^'], 0, '14'],
              ['14', actions['>'], 0, '15'],
              ['15', actions['>'], 1, '16']]
        trajectories.append(t)
        t = [['1', actions['>'], 0, '2'],
             ['2', actions['^'], 0, '6'],
             ['6', actions['^'], 0, '10'],
             ['10', actions['>'], 0, '11'],
             ['11', actions['^'], 0, '15'],
             ['15', actions['>'], 2, '16']]
        trajectories.append(t)

        t = [['1', actions['>'], 0, '2'],
             ['2', actions['^'], 0, '6'],
             ['6', actions['^'], 0, '10'],
             ['10', actions['^'], 0, '14'],
             ['14', actions['>'], 0, '15'],
             ['15', actions['v'], 0, '11'],
             ['11', actions['v'], 0, '7'],
             ['7', actions['<'], 0, '6'],
             ['6', actions['^'], 0, '10'],
             ['10', actions['>'], 0, '11'],
             ['11', actions['^'], 0, '15'],
             ['15', actions['>'], 2, '16']]
        trajectories.append(t)
        t = [['1', actions['>'], 0, '2'],
             ['2', actions['^'], 0, '6'],
             ['6', actions['^'], 0, '10'],
             ['10', actions['^'], 0, '14'],
             ['14', actions['>'], 0, '15'],
             ['15', actions['v'], 0, '11'],
             ['11', actions['v'], 0, '7'],
             ['7', actions['^'], 0, '11'],
             ['11', actions['^'], 0, '15'],
             ['15', actions['>'], 1, '16']]
        trajectories.append(t)
        AM = AbstractionMachine(trajectories)
        AM.resolve_reward_conflicts()

    def est_error(self):
        actions = {'^': '^',
                   'v': 'v',
                   '<': '<',
                   '>': '>',
                   'o': 'o'}
        trajectories = [[['1', actions['^'], 0, '4'],
                         ['4', actions['>'], 0, '5'],
                         ['5', actions['<'], 0, '4'],
                         ['4', actions['^'], 0, '7'],
                         ['7', actions['>'], 0, '8'],
                         ['8', actions['>'], 1, '9']],

                        [['1', actions['^'], 0, '4'],
                         ['4', actions['>'], 0, '5'],
                         ['5', actions['^'], 0, '8'],
                         ['8', actions['>'], 2, '9']],

                        [['1', actions['^'], 0, '4'],
                         ['4', actions['>'], 0, '5'],
                         ['5', actions['^'], 0, '8'],
                         ['8', actions['v'], 0, '5'],
                         ['5', actions['>'], 0, '6'],
                         ['6', actions['^'], 2, '9']],

                        [['1', actions['^'], 0, '4'],
                         ['4', actions['>'], 0, '5'],
                         ['5', actions['^'], 0, '8'],
                         ['8', actions['v'], 0, '5'],
                         ['5', actions['>'], 0, '6'],
                         ['6', actions['o'], 0, '6'],
                         ['6', actions['^'], 3, '9']],


                        [['1', actions['^'], 0, '4'],
                         ['4', actions['>'], 0, '5'],
                         ['5', actions['>'], 0, '6'],
                         ['6', actions['o'], 0, '6'],
                         ['6', actions['^'], 1, '9']]]
        AM = AbstractionMachine(trajectories)
        AM.resolve_reward_conflicts(write_file=True, make_graph=True)

    def est_add_trajectories(self):
        actions = {'^': '^',
                   'v': 'v',
                   '<': '<',
                   '>': '>',
                   'o': 'o'}
        t1 = [['1', actions['^'], 0, '4'],
              ['4', actions['>'], 0, '5'],
              ['5', actions['^'], 0, '8'],
              ['8', actions['v'], 0, '5'],
              ['5', actions['>'], 0, '6'],
              ['6', actions['^'], 2, '9']]
        t2 = [['1', actions['^'], 0, '4'],
              ['4', actions['>'], 0, '5'],
              ['5', actions['>'], 0, '6'],
              ['6', actions['o'], 0, '6'],
              ['6', actions['^'], 1, '9']]
        AM = AbstractionMachine([t1,t2], run_q_vals=True, action_set=['^', 'v', '<', '>', 'o'], verbose=True)
        AM.resolve_reward_conflicts(make_graph=True)
        t3 = [['1', actions['v'], 0, '5'],
              ['5', actions['>'], 0, '6'],
              ['6', actions['o'], 0, '6'],
              ['6', actions['^'], 0.5, '9']]
        AM.add_trajectory(t3, make_graph=True)

    def test_one_traj_at_a_time(self):
        actions = {'^': '^',
                   'v': 'v',
                   '<': '<',
                   '>': '>',
                   'o': 'o'}
        t1 = [['1', actions['^'], 0, '4'],
              ['4', actions['>'], 0, '5'],
              ['5', actions['^'], 0, '8'],
              ['8', actions['v'], 0, '5'],
              ['5', actions['>'], 0, '6'],
              ['6', actions['^'], 2, '9']]
        AM = AbstractionMachine(run_q_vals=True, action_set=['^', 'v', '<', '>', 'o'], verbose=True)
        AM.add_trajectory(t1)
        t2 = [['1', actions['^'], 0, '4'],
              ['4', actions['>'], 0, '5'],
              ['5', actions['>'], 0, '6'],
              ['6', actions['o'], 0, '6'],
              ['6', actions['^'], 1, '9']]
        AM.add_trajectory(t2)
        t3 = [['1', actions['v'], 0, '5'],
              ['5', actions['>'], 0, '6'],
              ['6', actions['o'], 0, '6'],
              ['6', actions['^'], 0.5, '9']]
        AM.add_trajectory(t3, make_graph=True)
        AM.reset()
        action = AM.get_action('1', eps=0)
        AM.step('1',action, 0, '5')
        action = AM.get_action('5')
        print(action)