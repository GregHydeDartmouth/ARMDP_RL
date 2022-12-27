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

    def est_one_traj_at_a_time(self):
        actions = {'^': 0,
                   'v': 1,
                   '<': 2,
                   '>': 3,
                   'o': 4}
        t1 = [['[1]', actions['^'], 0, '[4]'],
              ['[4]', actions['>'], 0, '[5]'],
              ['[5]', actions['^'], 0, '[8]'],
              ['[8]', actions['v'], 0, '[5]'],
              ['[5]', actions['>'], 0, '[6]'],
              ['[6]', actions['^'], 2, '[9]']]
        AM = AbstractionMachine(run_q_vals=True, action_set=[0, 1, 2, 3, 4], verbose=True)
        action = AM.get_action('[1]', eps=0)
        AM.add_trajectory(t1, resolve_non_zero= True, make_graph=True)
        t2 = [['[1]', actions['^'], 0, '[4]'],
              ['[4]', actions['>'], 0, '[5]'],
              ['[5]', actions['>'], 0, '[6]'],
              ['[6]', actions['o'], 0, '[6]'],
              ['[6]', actions['^'], 1, '[9]']]
        AM.add_trajectory(t2, make_graph=True)
        t3 = [['[1]', actions['v'], 1, '[5]'],
              ['[5]', actions['>'], 0, '[6]'],
              ['[6]', actions['o'], 0, '[6]'],
              ['[6]', actions['^'], 2, '[9]']]
        AM.add_trajectory(t3, make_graph=True)
        AM.reset()
        action = AM.get_action('[1]', eps=0)
        AM.step('[1]',action, 0, '[5]')
        action = AM.get_action('[5]')

    def est_state_granularity(self):
        t1 = [['1', '^', 0, '4'],
              ['4', '>', 0, '5'],
              ['5', '^', 0, '8'],
              ['8', 'v', 0, '5'],
              ['5', '>', 0, '6'],
              ['6', '^', 2, '9']]
        t2 = [['1', '^', 0, '4'],
              ['4', '>', 0, '5'],
              ['5', 'v', 0, '6'],
              ['6', '>', 0, '8'],
              ['8', 'v', 0, '7'],
              ['7', 'v', 2, '9']]
        t3 = [['1', '^', 0, '4'],
              ['4', '>', 0, '5'],
              ['5', '>', 0, '6'],
              ['6', 'o', 0, '6'],
              ['6', '^', 1, '9']]
        t4 = [['1', '^', 0, '4'],
              ['4', '>', 0, '5'],
              ['5', '>', 0, '6'],
              ['6', '<', 0, '7'],
              ['7', 'v', 1, '9']]
        AM = AbstractionMachine([t1, t2, t3, t4], granularity='state', run_q_vals=True, action_set=['^', 'v', '<', '>', 'o'], verbose=True)
        AM.resolve_reward_conflicts(make_graph=True)

    def test_only_solve_conflicting(self):
        AM = AbstractionMachine(granularity='triple', run_q_vals=True, action_set=['^', 'v', '<', '>', 'o'], verbose=True)
        t1 = [['1', '^', 0, '4'],
              ['4', '>', 0, '5'],
              ['5', '^', 0, '8'],
              ['8', 'v', 0, '5'],
              ['5', '>', 0, '6'],
              ['6', '^', 2, '9']]
        AM.add_trajectory(t1, resolve_non_zero=True, make_graph=True)
        t2 = [['1', '^', 0, '4'],
              ['4', '>', 0, '5'],
              ['5', 'v', 0, '6'],
              ['6', '>', 0, '8'],
              ['8', 'v', 0, '7'],
              ['7', 'v', 2, '9']]
        AM.add_trajectory(t2, resolve_non_zero=True, make_graph=True)
        t3 = [['1', '^', 0, '4'],
              ['4', '>', 0, '5'],
              ['5', '>', 0, '6'],
              ['6', 'o', 0, '6'],
              ['6', '^', 1, '9']]
        AM.add_trajectory(t3, resolve_non_zero=True, make_graph=True)