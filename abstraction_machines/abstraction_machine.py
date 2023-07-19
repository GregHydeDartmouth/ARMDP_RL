import graphviz
from gurobipy import *
from collections import defaultdict
import matplotlib.pyplot as plt

class AbstractionMachine():
    def __init__(self, conflicting_trajectories, granularity='state'):
        self.conflicting_trajectories = conflicting_trajectories
        self.granularity = granularity
        self.solution_set = None

    def make_cross_product_graph(self):
        if self.solution_set is not None:
            g = graphviz.Digraph('AMDP', format='png')
            for k, trajectory in enumerate(self.conflicting_trajectories):
                for l, triple in enumerate(trajectory):
                    state, action, reward, next_state = triple
                    if l == 0:
                        split_state = '{}^{}'.format(state, 0)
                    for i in range(self.depth):
                        if self.solution_set['traj_{}_triple_{}_next_state_{}_level_{}'.format(k, l, next_state, i)] == 1:
                            split_next_state = '{}^{}'.format(next_state, i)
                    g.node(split_state, shape='box')
                    g.node(split_next_state, shape='box')
                    g.edge(split_state, split_next_state, label='a={}/r={}'.format(action, reward))
                    split_state = split_next_state
            g.render(filename="graphs/AMDP", format="png")
                    
    def solve(self, depth = 2, min_obj = 0):
        self.depth = depth
        while True:
            conflict_resolver = Model("Conflict Resolver")

            jump_types, reward_jump_types = self._sum_to_one_constraint(conflict_resolver, depth)
            self._reward_mutex_constraint(reward_jump_types, conflict_resolver)
            change_level_indicators = self._transition_ambiguity_constraint(jump_types, conflict_resolver)

            # OBJECTIVE
            z = conflict_resolver.addVar(name='objective', vtype=GRB.INTEGER)
            conflict_resolver.addConstr(z == quicksum(change_level_indicators), name='objective_constr')
            conflict_resolver.addConstr(z >= min_obj, name='objective_floor_constr')
            conflict_resolver.setObjective(z, GRB.MINIMIZE)
            conflict_resolver.write('test.lp')
            conflict_resolver.optimize()
            if conflict_resolver.status == GRB.OPTIMAL:
                min_obj = conflict_resolver.objVal
                self.solution_set = dict()
                for v in conflict_resolver.getVars():
                    self.solution_set[v.varName] = v.x
                    if v.x == 1:
                        print(v.varName, v.x)
                break
            else:
                depth += 1
        return depth, min_obj

    def _sum_to_one_constraint(self, conflict_resolver, depth):

        jump_types = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        reward_jump_types = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

        for k, trajectory in enumerate(self.conflicting_trajectories):
            state_toggles = None
            for l, triple in enumerate(trajectory):
                state, action, reward, next_state = triple
                # ENTRY LEVEL CONSTR
                if state_toggles is None:
                    state_toggle = conflict_resolver.addVar(vtype=GRB.BINARY, name='traj_{}_triple_{}_state_{}_level_{}'.format(k, l, state, 0))
                    conflict_resolver.addConstr(state_toggle == 1, 'entry_level_constr')
                    state_toggles = [state_toggle]

                next_state_toggles = []
                for j in range(depth):
                    next_state_toggle = conflict_resolver.addVar(vtype=GRB.BINARY, name='traj_{}_triple_{}_next_state_{}_level_{}'.format(k, l, next_state, j))
                    next_state_toggles.append(next_state_toggle)

                for i, state_toggle in enumerate(state_toggles):
                    for j, next_state_toggle in enumerate(next_state_toggles):
                        if self.granularity == 'state':
                            jump_types['{}'.format(next_state)][i][j].append((state_toggle, next_state_toggle))
                            reward_jump_types['{}'.format(next_state)][i][j][reward].append((state_toggle, next_state_toggle))
                        elif self.granularity == 'triple':
                            jump_types['{},{},{}'.format(state, action, next_state)][i][j].append((state_toggle, next_state_toggle))
                            reward_jump_types['{},{},{}'.format(state, action, next_state)][i][j][reward].append((state_toggle, next_state_toggle))
                # SUM TO ONE CONSTRAINT
                conflict_resolver.addConstr(quicksum(next_state_toggles) == 1, name='sum_to_one_constr')
                state_toggles = next_state_toggles
        return jump_types, reward_jump_types
    
    def _reward_mutex_constraint(self, reward_jump_types, conflict_resolver):
        # REWARD MUTEX CONSTRAINT
        for jump_type, i_dict in reward_jump_types.items():
            for i, j_dict in i_dict.items():
                for j, reward_dict in j_dict.items():
                    if len(reward_dict.keys()) == 1:
                        continue
                    reward_toggles_types = []
                    for reward, toggle_type_pairs in reward_dict.items():
                        toggle_type = []
                        for toggle_type_pair in toggle_type_pairs:
                            toggle = conflict_resolver.addVar(vtype=GRB.BINARY)
                            conflict_resolver.addConstr(quicksum(toggle_type_pair) >= 1.6 - 100000 * (1-toggle))
                            conflict_resolver.addConstr(quicksum(toggle_type_pair) <= 1.4 + 100000 * (toggle))
                            toggle_type.append(toggle)
                        toggle_type_sum = conflict_resolver.addVar(vtype=GRB.BINARY)
                        conflict_resolver.addConstr(quicksum(toggle_type) >= 0.6 - 100000 * (1 - toggle_type_sum))
                        conflict_resolver.addConstr(quicksum(toggle_type) <= 0.4 + 100000 * (toggle_type_sum))
                        reward_toggles_types.append(toggle_type_sum)
                    conflict_resolver.addConstr(quicksum(reward_toggles_types) <= 1)

    def _transition_ambiguity_constraint(self, jump_types, conflict_resolver):
        # TRANSITION AMBIGUITY CONSTRAINT
        # also get level change indicators for objective
        change_level_indicators = []
        for jump_type, i_dict in jump_types.items():
            for i, j_dict in i_dict.items():
                transition_ambiguity_sum = []
                for j, toggle_type_pairs in j_dict.items():
                    toggle_type = []
                    for toggle_type_pair in toggle_type_pairs:
                        toggle = conflict_resolver.addVar(vtype=GRB.BINARY)
                        conflict_resolver.addConstr(quicksum(toggle_type_pair) >= 1.6 - 100000 * (1-toggle))
                        conflict_resolver.addConstr(quicksum(toggle_type_pair) <= 1.4 + 100000 * (toggle))
                        toggle_type.append(toggle)
                    toggle_sum = conflict_resolver.addVar(vtype=GRB.BINARY)
                    conflict_resolver.addConstr(quicksum(toggle_type) >= 0.6 - 100000 * (1 - toggle_sum))
                    conflict_resolver.addConstr(quicksum(toggle_type) <= 0.4 + 100000 * (toggle_sum))
                    if i != j:
                        change_level_indicators.append(toggle_sum)
                    transition_ambiguity_sum.append(toggle_sum)
                conflict_resolver.addConstr(quicksum(transition_ambiguity_sum) <= 1)
        return change_level_indicators


if __name__ == "__main__":
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
    AM = AbstractMachine(trajectories, granularity='state')
    AM.solve()
    AM.make_cross_product_graph()
