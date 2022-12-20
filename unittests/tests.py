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

    def test_error(self):
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

