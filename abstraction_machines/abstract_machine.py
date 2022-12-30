import json
import time
import random
from tqdm import tqdm
from gurobipy import *
from graphviz import Digraph
from collections import defaultdict

def _model_callback(model, where):
        if where == GRB.Callback.MIPNODE:
            obj = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
            # start timer at feasible solution
            if obj > 1e50:
                model._time = time.time()
            elif abs(obj - model._cur_obj) > 1e-8:
                model._cur_obj = obj
                model._time = time.time()
        if time.time() - model._time > model._change_tl:
            print("\nTerminated model due to timeout")
            model.terminate()

class AbstractionMachine():
    def __init__(self, trajectories = [], granularity='triple',num_threads=16, model_change_tl=None, verbose=False, run_q_vals=False, action_set=None, gamma=0.9):
        self.verbose = verbose
        self.num_threads = num_threads
        self.model_change_tl = model_change_tl
        self.run_q_vals = run_q_vals
        if self.run_q_vals:
            self.abstract_table = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
            self.abstract_Q_table = defaultdict(lambda: defaultdict(float))
            if action_set == None:
                raise Exception("can not run q_vals without action_size")
            else:
                self.action_set = [str(a) for a in action_set]
        self.gamma = gamma
        self.default_triple_set = set()
        self.exemplar_trajectories = trajectories
        self.granularity=granularity

        self.current_state = None
        self.depth = 2

    def _get_conflicting_trajectories(self):
        conflict_table = defaultdict(lambda: defaultdict(set))
        for i, traj in enumerate(self.exemplar_trajectories):
            for triple in traj:
                state, action, reward, next_state = triple
                if self.granularity =='triple':
                    conflict_table[(state, action, next_state)][reward].add(i)
                elif self.granularity=='state':
                    conflict_table[next_state][reward].add(i)

        conflicting_traj_idxs = set()
        for triple in conflict_table:
            if len(conflict_table[triple].keys()) > 1:
                if len(conflict_table[triple].keys()) > self.depth:
                    self.depth = len(conflict_table[triple].keys())
                    if self.verbose:
                        print("Depth insufficient to solve... increasing depth to max number of conflicting rewards for a triple. New Depth: {}".format(self.depth))
                for reward in conflict_table[triple]:
                    for traj_idx in conflict_table[triple][reward]:
                        conflicting_traj_idxs.add(traj_idx)
        conflicting_trajectories = []
        for conflict_traj_idx in conflicting_traj_idxs:
            conflicting_trajectories.append(self.exemplar_trajectories[conflict_traj_idx])
        return conflicting_trajectories

    def save_abstraction_model(self, file_name, save_q_vals = False):
        # abstract table in a list
        save_abstract_table = defaultdict(lambda: defaultdict(dict))
        for state in self.abstract_table:
            for action in self.abstract_table[state]:
                for next_state, reward_set in self.abstract_table[state][action].items():
                    save_abstract_table[state][action][next_state] = list(reward_set)
        model = {'abstraction': save_abstract_table}
        if save_q_vals:
            model['q_vals'] = self.abstract_Q_table
            #model.append(json.dumps(self.abstract_Q_table, indent=4))
        with open(file_name, 'w') as f:
            json.dump(model, f)

    def load_abstraction_model(self, file_name):
        with open(file_name, 'r') as f:
            model = json.load(f)
        self.abstract_table = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        for state in model['abstraction']:
            for action in model['abstraction'][state]:
                for next_state, reward_list in model['abstraction'][state][action].items():
                    self.abstract_table[state][action][next_state] = set(reward_list)
        if 'q_vals' in model.keys():
            self.abstract_Q_table = model['q_vals']

    def _get_transition_splits(self, drp, depth):
        choice_table = dict()
        choice_types = defaultdict(list)
        reward_table = dict()
        jump_contexts = []
        choice_mutex_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        reward_mutex_dict = defaultdict(lambda: defaultdict(list))

        variables = 0
        constraints = 0

        conflicting_trajectories = self._get_conflicting_trajectories()
        for k, traj in enumerate(tqdm(conflicting_trajectories, desc="Making split states...", disable=not self.verbose)):
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
                            if self.granularity == 'triple':
                                choice_types['({}^[{}],{},{}^[{}])'.format(state, i, action, next_state, j)].append(choice)
                            elif self.granularity == 'state':
                                choice_types['{}^[{}])'.format(next_state, j)].append(choice)
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
                                    if choice in other_choice_reward_dict:
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
        if self.verbose:
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
            if delta <= 0.0000001:
                break

    def _abstract_qsa_max(self, state):
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

    def _abstract_qsas(self, state):
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
            max_qsa, max_action = self._abstract_qsa_max(self.current_state)
            return max_action

    def get_state_policy(self, state):
        if self.current_state is None:
            self.current_state = '{}^[{}]'.format(state, 0)
        policy_mapping = self._abstract_qsas(self.current_state)
        return policy_mapping

    def update_abstractions(self, trajectory, add_trajectory=False, resolve_conflict = False, resolve_q_vals=False, find_feasible=False, write_file = False, make_graph = False):
        if add_trajectory:
            self.exemplar_trajectories.append(trajectory)
        if resolve_conflict:
            self.exemplar_trajectories.append(trajectory)
            self.resolve_reward_conflicts(find_feasible=find_feasible, write_file=write_file, make_graph=make_graph)
        elif resolve_q_vals:
            self.exemplar_trajectories.append(trajectory)
            self._solve_abstract_MDP()
            if make_graph:
                self._build_abstract_MDP_graph()

    def step(self, state, action, reward, next_state, evaluate=False):
        conflict = False
        new_q_val = False
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
                if not evaluate:
                    self.abstract_table[self.current_state][action][next_state].add(reward)
                if reward != 0:
                    # no actual conflict, just forcing a resolve if there is a new reward triple that changes q vals
                    new_q_val = True
                self.current_state = next_state
            else:
                if not evaluate:
                    self.abstract_table[self.current_state][action][some_next_state].add(reward)
                if len(self.abstract_table[self.current_state][action][some_next_state]) > 1:
                    conflict = True
                self.current_state = some_next_state
        else:
            current_level = self.current_state.split('^')[1]
            next_state = '{}^{}'.format(next_state, current_level)
            if not evaluate:
                self.abstract_table[self.current_state][action][next_state].add(reward)
            if reward != 0:
                # no actual conflict, just forcing a resolve if there is a new reward triple that changes q vals
                new_q_val = True
            self.current_state = next_state
        return conflict, new_q_val

    def reset(self):
        self.current_state = None

    def resolve_reward_conflicts(self, find_feasible=False, write_file=False, make_graph=False):
        if self.verbose:
            print("\nMaking Dual Reward Problem")
            print("Model Depth: {}".format(self.depth))

        while True:
            drp = Model("Dual Reward Problem")
            #drp.setParam('MIPFocus',1)
            drp.setParam('Threads',self.num_threads)
            if find_feasible:
                drp.setParam('SolutionLimit',1)
            # get split states
            reward_table, jump_contexts = self._get_transition_splits(drp, self.depth)
            z = drp.addVar(name='objective', vtype=GRB.INTEGER)
            drp.addConstr(z == quicksum(jump_contexts), name='objective_constr')
            drp.setObjective(z, GRB.MINIMIZE)
            if write_file:
                drp.write('model_with_depth={}.lp'.format(self.depth))
            if self.model_change_tl is not None:
                drp._cur_obj=float('inf')
                drp._time = time.time()
                drp._change_tl = self.model_change_tl
                drp.optimize(callback=_model_callback)
            else:
                drp.optimize()
            var_dict = dict()
            try:
                for v in drp.getVars():
                    var_dict[v.varName] = v.x
            except:
                self.depth += 1
                print("Depth insufficient to solve... increasing depth by 1. New Depth: {}".format(self.depth))
                continue
            # build abstract MDP
            self._build_abstract_MDP(var_dict, reward_table)
            # add non-conflicting trajs to the abstract MDP
            for traj in self.exemplar_trajectories:
                self.reset()
                for triple in traj:
                    state, action, reward, next_state = triple
                    self.step(state,action,reward,next_state)
            self.reset()

            if self.run_q_vals:
                self._solve_abstract_MDP()
            if make_graph:
                self._build_abstract_MDP_graph()
            break
        return

    def _build_abstract_MDP(self, var_dict, reward_table):
        self.abstract_table = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        for variable, value in var_dict.items():
            if '_choice' in variable and value > 0.5:
                reward = reward_table[variable]
                triple = variable.split('(')[1].split(')')[0].split(',')
                self.abstract_table[triple[0]][triple[1]][triple[2]].add(reward)

    def _build_abstract_MDP_graph(self):
        g = Digraph('abstract_MDP')
        for state in self.abstract_table:
            for action in self.abstract_table[state]:
                if self.run_q_vals:
                    q_sa = round(self.abstract_Q_table[state][action], 2)
                for next_state in self.abstract_table[state][action]:
                    for reward in self.abstract_table[state][action][next_state]:
                        g.node(state)
                        g.node(next_state)
                        if self.run_q_vals:
                            g.edge(state, next_state, label='a={},r={},q_sa={}'.format(action, reward, q_sa))
                        else:
                            g.edge(state, next_state, label='a={},r={}'.format(action, reward))
        g.render('graphs/abstract_MDP', view=False)
