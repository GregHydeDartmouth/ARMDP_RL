import os
from tqdm import tqdm
from gurobipy import *
from graphviz import Digraph
from collections import defaultdict

class AbstractionMachine():
    def __init__(self, trajectories, verbose=False):
        self.verbose = verbose
        self.action_mapping, self.state_mapping, self.exemplar_trajectories = \
            self._make_mappings(trajectories)

    def _make_mappings(self, trajectories):
        action_mapping = dict()
        state_mapping = dict()
        mapped_trajectories = []
        state_num = 0
        action_num = 0
        for traj in trajectories:
            mapped_trajectory = []
            for i, triple in enumerate(traj):
                state, action, reward, next_state = triple
                if i == 0:
                    if str(state) not in state_mapping:
                        state_mapping[str(state)] = str(state_num)
                        state_num += 1
                if str(action) not in action_mapping:
                    action_mapping[str(action)] = str(action_num)
                    action_num += 1
                if str(next_state) not in state_mapping:
                    state_mapping[str(next_state)] = str(state_num)
                    state_num += 1
                new_state = state_mapping[str(state)]
                new_action = action_mapping[str(action)]
                new_next_state = state_mapping[str(next_state)]
                mapped_trajectory.append([new_state, new_action, reward,new_next_state])
            mapped_trajectories.append(mapped_trajectory)
        return action_mapping, state_mapping, mapped_trajectories

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

    def resolve_reward_conflicts(self, depth=2, write_file=False, make_graph=False):
        print("\nMaking Dual Reward Problem")

        while True:
            drp = Model("Dual Reward Problem")
            # get split states
            reward_table, jump_contexts = self._get_transition_splits(drp, depth)
            z = drp.addVar(name='objective', vtype=GRB.INTEGER)
            drp.addConstr(z == quicksum(jump_contexts), name='objective_constr')
            drp.setObjective(z, GRB.MINIMIZE)
            if write_file:
                drp.write('model_with_depth={}.lp'.format(depth))
                self._write_mappings()
            drp.optimize()
            var_dict = dict()
            try:
                for v in drp.getVars():
                    var_dict[v.varName] = v.x
            except:
                depth += 1
                print("Depth insufficient to solve... increasing depth by 1. New Depth: {}".format(depth))
                continue
            abstract_table = self.build_abstract_MDP(var_dict, reward_table)
            if make_graph:
                self.build_hierarchical_MDP_graph(var_dict, reward_table)
            break
        return abstract_table

    def build_abstract_MDP(self, var_dict, reward_table):
        abstract_table = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        for variable, value in var_dict.items():
            if '_choice' in variable and value > 0.5:
                reward = reward_table[variable]
                triple = variable.split('(')[1].split(')')[0].split(',')
                abstract_table[triple[0]][triple[1]][triple[2]].add(reward)
        return abstract_table

    def build_hierarchical_MDP_graph(self, var_dict, reward_table):
        g = Digraph('hierarchical_MDP')
        edges = set()
        for variable, value in var_dict.items():
            if '_choice' in variable and value > 0.5:
                reward = reward_table[variable]
                triple = variable.split('(')[1].split(')')[0].split(',')
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
                    g.edge(triple[0], triple[2], label='a={},r={}'.format(triple[1], reward))
                    edges.add(tuple(triple))
        g.render('graphs/hierarchical_MDP', view=False)