import json
import time
import copy
import random
from tqdm import tqdm
from gurobipy import *
from graphviz import Digraph
from scipy.special import softmax
from collections import defaultdict

class AbstractionMachine():
    def __init__(self, action_set, gamma=0.9, granularity='triple', num_threads=16, verbose=False, make_graphs=False, write_files=False):
        self.verbose = verbose
        self.num_threads = num_threads
        self.make_graphs = make_graphs
        self.write_files = write_files

        # initialize tables assuming normal MDP
        self.abstract_table = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self.abstract_Q_table = defaultdict(lambda: defaultdict(float))

        # action space and discount
        self.action_set = [str(a) for a in action_set]
        self.gamma = gamma

        # holds all trajectories experienced
        self.exemplar_trajectories = []
        # holds subset of trajectories known to conflict for current AMDP
        self.conflicting_trajectories = []

        # table to see if trajectory conflicts somewhere
        self.reward_conflicts_table = defaultdict(lambda:defaultdict(int))
        self.granularity=granularity

        self.current_state = None
        self.trajectory_trace = []

        self.depth = 2
        self.min_obj = 0

    def save_abstraction_model(self, file_name):
        # abstract table in a list
        save_abstract_table = defaultdict(lambda: defaultdict(dict))
        for state in self.abstract_table:
            for action in self.abstract_table[state]:
                for next_state, reward_set in self.abstract_table[state][action].items():
                    save_abstract_table[state][action][next_state] = list(reward_set)
        model = {'abstraction': save_abstract_table}
        model['q_vals'] = self.abstract_Q_table
        with open(file_name, 'w') as f:
            json.dump(model, f)

    def load_abstraction_model(self, file_name):
        with open(file_name, 'r') as f:
            model = json.load(f)
        # prepend
        self.abstract_tables = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        for state in model['abstraction']:
            for action in model['abstraction'][state]:
                for next_state, reward_list in model['abstraction'][state][action].items():
                    self.abstract_table[state][action][next_state] = set(reward_list)
        if 'q_vals' in model.keys():
            # prepend
            self.abstract_Q_table = model['q_vals']

    def abstract_qsa_max(self, state):
        max_qsa = None
        max_action = None
        equivalent_actions = []
        if state in self.abstract_Q_table:
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
        else:
            max_action = random.choice(self.action_set)
            max_qsa = 0
        return max_qsa, max_action

    def abstract_qsas(self, state):
        qsas = dict()
        for action in self.action_set:
            val = round(self.abstract_Q_table[state][action], 5)
            qsas[action] = val
        return qsas

    def get_action(self, state, eps = 0.1):
        if random.random() < eps:
            action = random.choice(self.action_set)
            return action
        else:
            if self.current_state is None:
                self.current_state = '{}^[{}]'.format(state, 0)
            max_qsa, max_action = self.abstract_qsa_max(self.current_state)
            return max_action

    def get_softmax_action(self, state):
        if self.current_state is None:
            self.current_state = '{}^[{}]'.format(state, 0)
        q_vals = self.abstract_qsas(self.current_state)
        actions = []
        vals = []
        for action, val in q_vals.items():
            actions.append(action)
            vals.append(val)
        distr = softmax(vals)
        action = random.choices(actions, distr, k=1)[0]
        return action



    def get_state_policy(self, state):
        if self.current_state is None:
            self.current_state = '{}^[{}]'.format(state, 0)
        policy_mapping = self.abstract_qsas(self.current_state)
        return policy_mapping

    def update_reward_conflicts_table(self):
        conflict = False
        triple_in = set()
        for triple in self.trajectory_trace:
            state, action, reward, next_state = triple
            #print(state, action, reward, next_state)
            if (state, action, reward, next_state) not in triple_in:
                if (state, action, next_state) in self.reward_conflicts_table:
                    #print(self.reward_conflicts_table[(state, action, next_state)])
                    if reward in self.reward_conflicts_table[(state, action, next_state)]:
                        self.reward_conflicts_table[(state, action, next_state)][reward].append(len(self.exemplar_trajectories)-1)
                        if len(self.reward_conflicts_table[(state, action, next_state)]) > 1:
                            conflict = True
                    else:
                        self.reward_conflicts_table[(state, action, next_state)][reward] = [len(self.exemplar_trajectories)-1]
                        if len(self.reward_conflicts_table[(state, action, next_state)]) >= 1:
                            conflict = True
                else:
                    self.reward_conflicts_table[(state, action, next_state)][reward] = [len(self.exemplar_trajectories)-1]
                triple_in.add((state, action, reward, next_state))
        return conflict

    def initialize_conflicting_trajectories(self):
        for triple, reward_dict in self.reward_conflicts_table.items():
            if len(reward_dict.keys()) > 1:
                for reward, exemplar_ids in reward_dict.items():
                    exemplar_id = random.choice(exemplar_ids)
                    self.conflicting_trajectories.append(self.exemplar_trajectories[exemplar_id])
                break
        self.old_conflicting_trajectories = copy.deepcopy(self.conflicting_trajectories)

    def balance_evidence(self, state, action, reward, next_state):
        for _reward, exemplar_ids in self.reward_conflicts_table[(state, action, next_state)].items():
            if reward != _reward:
                exemplar_id = random.choice(exemplar_ids)
                self.conflicting_trajectories.append(self.exemplar_trajectories[exemplar_id])

    def update_abstractions(self):
        self.exemplar_trajectories.append(self.trajectory_trace)
        conflict = self.update_reward_conflicts_table()
        if conflict:
            if self.conflicting_trajectories == []:
                self.initialize_conflicting_trajectories()
                self.resolve_reward_conflicts()
            else:
                self.current_state = None
                for trip in self.trajectory_trace:
                    state, action, reward, next_state = trip
                    AMDP_conflict, run_q_vals = self.step(state, action, reward, next_state, update=True)
                    if AMDP_conflict:
                        self.conflicting_trajectories.append(self.trajectory_trace)
                        #self.balance_evidence(state, action, reward, next_state)
                        self.resolve_reward_conflicts()
                        break
        else:
            self.current_state = None
            new_nonzero_reward = False
            for trip in self.trajectory_trace:
                state, action, reward, next_state = trip
                AMDP_conflict, run_q_vals = self.step(state, action, reward, next_state, update=True)
                assert not AMDP_conflict, 'Error in AMDP. Trajectory previously determined to not be conflicting'
                if not new_nonzero_reward and run_q_vals:
                    new_nonzero_reward = True
            if new_nonzero_reward:
                self.solve_abstract_MDP()
                #self.build_abstract_MDP_graph()

    def step(self, state, action, reward, next_state, update=False):
        if not update:
            self.trajectory_trace.append((state, action, reward, next_state))
        conflict = False
        run_q_vals = False
        if self.current_state is None:
            self.current_state = '{}^[0]'.format(state)
        if self.current_state in self.abstract_table:
            if action in self.abstract_table[self.current_state]:
                abstract_next_state = None
                for some_next_state in self.abstract_table[self.current_state][action]:
                    if next_state == some_next_state.split('^')[0]:
                        abstract_next_state = some_next_state
                        break
                if abstract_next_state is not None:
                    if reward in self.abstract_table[self.current_state][action][abstract_next_state]:
                        self.current_state = abstract_next_state
                        return conflict, run_q_vals
                    else:
                        self.current_state = abstract_next_state
                        conflict = True
                        return conflict, run_q_vals
        # fell off
        current_level = self.current_state.split('^')[1]
        abstract_next_state = '{}^{}'.format(next_state, current_level)
        if update:
            self.abstract_table[self.current_state][action][abstract_next_state].add(reward)
            assert len(self.abstract_table[self.current_state][action][abstract_next_state]) == 1, "something wrong"
        if reward != 0:
            run_q_vals = True
        self.current_state = abstract_next_state

        return conflict, run_q_vals

    def reset(self):
        if self.trajectory_trace != []:
            self.update_abstractions()
        self.trajectory_trace = []
        self.current_state = None


    def build_abstract_MDP(self, var_dict, reward_table):
        self.abstract_table = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        for variable, value in var_dict.items():
            if '_choice' in variable and value > 0.5:
                reward = reward_table[variable]
                triple = variable.split('(')[1].split(')')[0].split(',')
                self.abstract_table[triple[0]][triple[1]][triple[2]].add(reward)

    def build_abstract_MDP_graph(self):
        g = Digraph('abstract_MDP')
        for state in self.abstract_table:
            for action in self.abstract_table[state]:
                q_sa = round(self.abstract_Q_table[state][action], 2)
                for next_state in self.abstract_table[state][action]:
                    for reward in self.abstract_table[state][action][next_state]:
                        g.node(state)
                        g.node(next_state)
                        g.edge(state, next_state, label='a={},r={},q_sa={}'.format(action, reward, q_sa))
        g.render('graphs/abstract_MDP', view=False)

    def resolve_reward_conflicts(self):
        if self.verbose:
            print("\nMaking Dual Reward Problem")
            print("Model Depth: {}".format(self.depth))

        while True:
            drp = Model("Dual Reward Problem")
            #drp.setParam('MIPFocus',1)
            drp.setParam('Threads',self.num_threads)
            # get split states
            reward_table, jump_contexts = self.get_transition_splits(drp, self.depth)
            z = drp.addVar(name='objective', vtype=GRB.INTEGER)
            drp.addConstr(z == quicksum(jump_contexts), name='objective_constr')
            drp.addConstr(z >= self.min_obj, name='objective_floor_constr')
            drp.setObjective(z, GRB.MINIMIZE)
            if self.write_files:
                drp.write('model_with_depth={}.lp'.format(self.depth))
            drp.optimize()
            var_dict = dict()
            try:
                for v in drp.getVars():
                    var_dict[v.varName] = v.x
                    if v.varName == 'objective':
                        self.min_obj = v.x
            except:
                self.depth += 1
                self.min_obj = 0
                self.conflicting_trajectories = copy.deepcopy(self.old_conflicting_trajectories)
                print("Depth insufficient to solve... increasing depth by 1. New Depth: {}".format(self.depth))
                continue
            # build abstract MDP
            self.build_abstract_MDP(var_dict, reward_table)
            self.solve_abstract_MDP()

            if self.make_graphs:
                self.build_abstract_MDP_graph()
            break
        return

    def solve_abstract_MDP(self, tf=None):
        self.abstract_Q_table = defaultdict(lambda: defaultdict(float))
        for state in self.abstract_table:
            for action in self.action_set:
                self.abstract_Q_table[state][action] = 0
        if self.verbose:
            print('solving abstract MDP')
        while True:
            delta = 0
            for state, action_dict in self.abstract_table.items():
                for action, next_state_dict in action_dict.items():
                    qsa = 0
                    if tf is None:
                        if len(next_state_dict.keys()) > 1:
                            print(state, action, next_state_dict.keys())
                        assert len(list(next_state_dict.keys())) == 1, 'There is an ambiguity'
                    for next_state, rewards in next_state_dict.items():
                        reward_vals = list(rewards)
                        assert len(reward_vals) == 1, 'something wrong with rewards'
                        if tf is not None:
                            probability = tf[state.split('^')[0]][action][next_state.split('^')[0]]
                        else:
                            probability = 1
                        next_state_qsa, next_state_qsa_action = self.abstract_qsa_max(next_state)
                        qsa += probability * reward_vals[0] + self.gamma * next_state_qsa
                    old_qsa = self.abstract_Q_table[state][action]
                    delta += abs(old_qsa - qsa)
                    self.abstract_Q_table[state][action] = qsa
            if self.verbose:
                print('Delta: {}'.format(delta))
            if delta <= 0.0000001:
                break

    def get_transition_splits(self, drp, depth):
        choice_table = dict()
        choice_types = defaultdict(list)
        reward_table = dict()
        jump_contexts = []
        choice_mutex_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        reward_mutex_dict = defaultdict(lambda: defaultdict(list))

        variables = 0
        constraints = 0

        for k, traj in enumerate(tqdm(self.conflicting_trajectories, desc="Making split states...", disable=not self.verbose)):
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
                    for j in range(i, depth):
                        choice = drp.addVar(name='traj_{}_triple_{}_({}^[{}],{},{}^[{}])_choice'.format(k, l, state, i, action, next_state, j), vtype=GRB.BINARY)
                        variables += 1
                        choices.append(choice)
                        choices_by_state[i].append(choice)
                        choices_by_next_state[j].append(choice)
                        if self.granularity == 'triple':
                            choice_mutex_dict['({}^[{}],{},{}^[{}])'.format(state, '', action, next_state, '')][i][j].append(choice)
                        elif self.granularity == 'state':
                            choice_mutex_dict['{}'.format(next_state)][i][j].append(choice)
                        if i != j:
                            if self.granularity == 'triple':
                                choice_types['({}^[{}],{},{}^[{}])'.format(state, i, action, next_state, j)].append(choice)
                            elif self.granularity == 'state':
                                choice_types['{}^{}'.format(next_state, j)].append(choice)
                        choice_table['traj_{}_triple_{}_({}^[{}],{},{}^[{}])_choice'.format(k, l, state, i, action, next_state, j)] = choice
                        reward_table['traj_{}_triple_{}_({}^[{}],{},{}^[{}])_choice'.format(k, l, state, i, action, next_state, j)] = reward
                        reward_mutex_dict['{}^[{}]'.format(next_state, j)][reward].append(choice)

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

        # ensure mutual exclusivity over conflicting reward states
        for state_j, reward_dict in reward_mutex_dict.items():
            for reward, choices in reward_dict.items():
                for _reward, _choices in reward_dict.items():
                    if reward != _reward:
                        for choice in choices:
                            for _choice in _choices:
                                drp.addConstr(choice + _choice, GRB.LESS_EQUAL, 1, name='reward_mutex_constr')

        for choice_type, choices in choice_types.items():
            choice_toggle = drp.addVar(name='{}_jump_toggle'.format(choice_type), vtype=GRB.BINARY)
            drp.addGenConstrIndicator(choice_toggle, False, quicksum(choices), GRB.EQUAL, 0)
            #drp.addGenConstrIndicator(choice_toggle, True, quicksum(choices), GRB.GREATER_EQUAL,1)
            jump_contexts.append(choice_toggle)

        if self.verbose:
            print('State splitting variables: {}, contstraints: {}'.format(variables, constraints))

        return reward_table, jump_contexts
