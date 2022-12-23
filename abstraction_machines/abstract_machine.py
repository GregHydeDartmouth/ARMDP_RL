import random
from tqdm import tqdm
from gurobipy import *
from graphviz import Digraph
from collections import defaultdict

class AbstractionMachine():
    def __init__(self, trajectories = [], verbose=False, run_q_vals=False, action_set=None, gamma=0.9):
        self.verbose = verbose
        self.run_q_vals = run_q_vals
        if self.run_q_vals:
            self.abstract_table = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
            self.abstract_Q_table = defaultdict(lambda: defaultdict(float))
            if action_set == None:
                raise Exception("can not run q_vals without action_size")
            else:
                self.action_set = action_set
        self.gamma = gamma
        #self.action_mapping = dict()
        #self.state_mapping = dict()
        #self.state_num = 0
        #self.action_num = 0
        self.default_triple_set = set()
        self.exemplar_trajectories = trajectories
        self.current_state = None
        self.depth = 2

    '''
    def _make_mappings(self, trajectories):
        mapped_trajectories = []
        for traj in trajectories:
            mapped_trajectory = []
            for i, triple in enumerate(traj):
                state, action, reward, next_state = triple
                if i == 0:
                    if str(state) not in self.state_mapping:
                        self.state_mapping[str(state)] = str(self.state_num)
                        self.state_mapping[str(self.state_num)] = str(state)
                        self.state_num += 1
                if str(action) not in self.action_mapping:
                    self.action_mapping[str(action)] = str(self.action_num)
                    self.action_mapping[str(self.action_num)] = str(action)
                    self.action_num += 1
                if str(next_state) not in self.state_mapping:
                    self.state_mapping[str(next_state)] = str(self.state_num)
                    self.state_mapping[str(self.state_num)] = str(next_state)
                    self.state_num += 1
                new_state = self.state_mapping[str(state)]
                new_action = self.action_mapping[str(action)]
                new_next_state = self.state_mapping[str(next_state)]
                mapped_trajectory.append([new_state, new_action, reward,new_next_state])
            mapped_trajectories.append(mapped_trajectory)
        return mapped_trajectories
    '''

    def _write_mappings(self):
        actions_file = open('action_mapping.txt', 'w')
        for action, mapping in self.action_mapping.items():
            actions_file.write('{}->{}\n'.format(action, mapping))
        actions_file.close()
        states_file = open('state_mapping.txt', 'w')
        for state, mapping in self.state_mapping.items():
            states_file.write('{}->{}\n'.format(state, mapping))
        states_file.close()

    def _get_transition_splits(self, drp, depth):
        choice_table = dict()
        choice_types = defaultdict(list)
        reward_table = dict()
        jump_contexts = []
        choice_mutex_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        reward_mutex_dict = defaultdict(lambda: defaultdict(list))

        variables = 0
        constraints = 0

        for k, traj in enumerate(tqdm(self.exemplar_trajectories, desc="Making split states...", disable=not self.verbose)):
            prev_choices_by_next_state = defaultdict(list)
            for l, triple in enumerate(traj):
                state, action, reward, next_state = triple

                choices = []
                choices_by_state = defaultdict(list)
                choices_by_next_state = defaultdict(list)
                reward_choice_dict = dict()
                for i in range(0, depth):
                    # first state level of any trajectory must 0
                    if l == 0 and i == 1:
                        break
                    for j in range(0, depth):
                        choice = drp.addVar(name='traj_{}_triple_{}_({}^[{}],{},{}^[{}])_choice'.format(k, l, state, i, action, next_state, j), vtype=GRB.BINARY)
                        variables += 1
                        choices.append(choice)
                        choices_by_state[i].append(choice)
                        choices_by_next_state[j].append(choice)
                        choice_mutex_dict['({}^[{}],{},{}^[{}])'.format(state, '', action, next_state, '')][i][j].append(choice)
                        if i != j:
                            choice_types['({}^[{}],{},{}^[{}])'.format(state, i, action, next_state, j)].append(choice)
                        choice_table['traj_{}_triple_{}_({}^[{}],{},{}^[{}])_choice'.format(k, l, state, i, action, next_state, j)] = choice
                        reward_table['traj_{}_triple_{}_({}^[{}],{},{}^[{}])_choice'.format(k, l, state, i, action, next_state, j)] = reward
                        reward_choice_dict['{},{}'.format(i,j)] = choice
                reward_mutex_dict['({},{},{})'.format(state, action, next_state)][reward].append(reward_choice_dict)
                drp.addConstr(quicksum(choices) == 1, name='choice_constr')
                constraints += 1

                # Trajectory continuity
                for prev_choice_level, prev_levels in prev_choices_by_next_state.items():
                    levels = choices_by_state[prev_choice_level]
                    drp.addConstr(quicksum(prev_levels) == quicksum(levels), name='context_constraint_j==i\'')
                    constraints += 1
                prev_choices_by_next_state = choices_by_next_state

        for mutex_constraint, i_dict in choice_mutex_dict.items():
            for i, j_dict in i_dict.items():
                outcome_toggles = []
                for j, choices in j_dict.items():
                    level_mutex_indicator = drp.addVar(name=mutex_constraint + '{},{}'.format(i, j), vtype=GRB.BINARY)
                    variables += 1
                    drp.addConstr(quicksum(choices) >= 0.6 - 100000 * (1 - level_mutex_indicator))
                    constraints += 1
                    drp.addConstr(quicksum(choices) <= 0.4 + 100000 * (level_mutex_indicator))
                    constraints += 1
                    outcome_toggles.append(level_mutex_indicator)
                drp.addConstr(quicksum(outcome_toggles) <= 1)
                constraints += 1

        # ensure mutual exclusivity over conflicting reward triples
        for triple, reward_dict in reward_mutex_dict.items():
            for reward, choice_reward_dict_list in reward_dict.items():
                for choice_reward_dict in choice_reward_dict_list:
                    for other_reward, other_choice_reward_dict_list in reward_dict.items():
                        if reward != other_reward:
                            for other_choice_reward_dict in other_choice_reward_dict_list:
                                for choice, choice_var in choice_reward_dict.items():
                                    drp.addConstr(choice_var + other_choice_reward_dict[choice], GRB.LESS_EQUAL, 1, name='reward_mutex_constr')


        for choice_type, choices in choice_types.items():
            choice_toggle = drp.addVar(name='{}_jump_toggle'.format(choice_type), vtype=GRB.BINARY)
            drp.addGenConstrIndicator(choice_toggle, False, quicksum(choices), GRB.EQUAL, 0)
            jump_contexts.append(choice_toggle)

        if self.verbose:
            print('State splitting variables: {}, contstraints: {}'.format(variables, constraints))

        return reward_table, jump_contexts

    def _solve_abstract_MDP(self, tf=None):
        self.abstract_Q_table = defaultdict(lambda: defaultdict(float))
        for state in self.abstract_table:
            for action in self.action_set:
                self.abstract_Q_table[state][action] = 0
        print('solving abstract MDP')
        while True:
            delta = 0
            for state, action_dict in self.abstract_table.items():
                for action, next_state_dict in action_dict.items():
                    qsa = 0
                    for next_state, rewards in next_state_dict.items():
                        reward_vals = list(rewards)
                        assert len(reward_vals) == 1, 'something wrong with rewards'
                        if tf is not None:
                            probability = tf[state.split('^')[0]][action][next_state.split('^')[0]]
                        else:
                            probability = 1
                        next_state_qsa, next_state_qsa_action = self._abstract_qsa_max(next_state)
                        qsa += probability * reward_vals[0] + self.gamma * next_state_qsa
                    old_qsa = self.abstract_Q_table[state][action]
                    delta += abs(old_qsa - qsa)
                    self.abstract_Q_table[state][action] = qsa
            if self.verbose:
                print('Delta: {}'.format(delta))
            if delta <= 1e-2:
                break

    def _abstract_qsa_max(self, state):
        max_qsa = None
        max_action = None
        equivalent_actions = []
        for action in self.action_set:
            val = round(self.abstract_Q_table[state][action], 5)
            if max_qsa is None:
                max_qsa = val
                max_action = action
                equivalent_actions = [(max_qsa, max_action)]
            else:
                if val > max_qsa:
                    max_qsa = val
                    max_action = action
                    equivalent_actions = [(max_qsa, max_action)]
                elif val == max_qsa:
                    equivalent_actions.append((val, action))
        if len(equivalent_actions) > 1:
            max_qsa, max_action = random.choice(equivalent_actions)
        return max_qsa, max_action

    def get_action(self, state, eps = 0.1):
        if random.random() < eps:
            action = random.choice(self.action_set)
            return action
        else:
            if self.current_state is None:
                self.current_state = '{}^[{}]'.format(state, 0)
            max_qsa, max_action = self._abstract_qsa_max(self.current_state)
            return max_action

    def add_trajectory(self, trajectory, write_file = False, make_graph = False):
        resolve = False
        add_traj = False
        hold_current_state = self.current_state
        self.current_state = '{}^[{}]'.format(trajectory[0][0], 0)
        for triple in trajectory:
            if str(triple) not in self.default_triple_set:
                self.default_triple_set.add(str(triple))
                add_traj = True
            state, action, reward, next_state = triple
            conflict = self.step(state, action, reward, next_state)
            if conflict:
                resolve = True
                add_traj = True
                break
        if add_traj:
            self.exemplar_trajectories.append(trajectory)
        if resolve:
            self.resolve_reward_conflicts(write_file=write_file, make_graph=make_graph)
        self.current_state = hold_current_state

    def step(self, state, action, reward, next_state):
        conflict = False
        if self.current_state is None:
            self.current_state = '{}^[{}]'.format(state, 0)
        if action in self.abstract_table[self.current_state]:
            found_abstract_state = None
            for some_next_state in self.abstract_table[self.current_state][action]:
                if some_next_state.split('^')[0] == next_state:
                    found_abstract_state = some_next_state
                    break
            if found_abstract_state is None:
                current_level = self.current_state.split('^')[1]
                next_state = '{}^{}'.format(next_state, current_level)
                self.abstract_table[self.current_state][action][next_state].add(reward)
                self.current_state = next_state
            else:
                self.abstract_table[self.current_state][action][some_next_state].add(reward)
                if len(self.abstract_table[self.current_state][action][some_next_state]) > 1:
                    conflict = True
                self.current_state = some_next_state
        else:
            current_level = self.current_state.split('^')[1]
            next_state = '{}^{}'.format(next_state, current_level)
            self.abstract_table[self.current_state][action][next_state].add(reward)
            self.current_state = next_state
        return conflict

    def reset(self):
        self.current_state = None

    def resolve_reward_conflicts(self, write_file=False, make_graph=False):
        print("\nMaking Dual Reward Problem")

        while True:
            drp = Model("Dual Reward Problem")
            # get split states
            reward_table, jump_contexts = self._get_transition_splits(drp, self.depth)
            z = drp.addVar(name='objective', vtype=GRB.INTEGER)
            drp.addConstr(z == quicksum(jump_contexts), name='objective_constr')
            drp.setObjective(z, GRB.MINIMIZE)
            if write_file:
                drp.write('model_with_depth={}.lp'.format(self.depth))
            drp.optimize()
            var_dict = dict()
            try:
                for v in drp.getVars():
                    var_dict[v.varName] = v.x
            except:
                self.depth += 1
                print("Depth insufficient to solve... increasing depth by 1. New Depth: {}".format(self.depth))
                continue
            self.build_abstract_MDP(var_dict, reward_table)
            if self.run_q_vals:
                self._solve_abstract_MDP()
            if make_graph:
                self.build_abstract_MDP_graph(var_dict, reward_table)
            break
        return

    def build_abstract_MDP(self, var_dict, reward_table):
        self.abstract_table = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        for variable, value in var_dict.items():
            if '_choice' in variable and value > 0.5:
                reward = reward_table[variable]
                triple = variable.split('(')[1].split(')')[0].split(',')
                self.abstract_table[triple[0]][triple[1]][triple[2]].add(reward)

    def build_abstract_MDP_graph(self, var_dict, reward_table):
        g = Digraph('abstract_MDP')
        edges = set()
        for variable, value in var_dict.items():
            if '_choice' in variable and value > 0.5:
                reward = reward_table[variable]
                triple = variable.split('(')[1].split(')')[0].split(',')
                if self.run_q_vals:
                    q_sa = round(self.abstract_Q_table[triple[0]][triple[1]],2)
                if tuple(triple) not in edges:
                    try:
                        scale = 4
                        level_scale = 20
                        state_location = triple[0].split(' ')
                        state_i = int(state_location[1].split('^')[1][1])
                        state_from = (int(state_location[0][1]) * scale + (level_scale * state_i), int(state_location[1][0]) * scale + (level_scale * state_i))
                        next_state_location = triple[2].split(' ')
                        next_state_j = int(next_state_location[1].split('^')[1][1])
                        next_state_to = (int(next_state_location[0][1]) * scale + (level_scale * next_state_j), int(next_state_location[1][0]) * scale + (level_scale * next_state_j))
                        g.node(triple[0], pos='{},{}!'.format(state_from[0], state_from[1]))
                        g.node(triple[2], pos='{},{}!'.format(next_state_to[0], next_state_to[1]))
                    except:
                        g.node(triple[0])
                        g.node(triple[2])
                    if self.run_q_vals:
                        g.edge(triple[0], triple[2], label='a={},r={},q_sa={}'.format(triple[1], reward, q_sa))
                    else:
                        g.edge(triple[0], triple[2], label='a={},r={}'.format(triple[1], reward))
                    edges.add(tuple(triple))
        g.render('graphs/abstract_MDP', view=False)