import graphviz
from gurobipy import *
from collections import defaultdict
import matplotlib.pyplot as plt

class AbstractionMachine():
    def __init__(self, conflicting_trajectories, granularity='state', monotonic_levels = False):
        self.conflicting_trajectories = conflicting_trajectories
        self.granularity = granularity
        self.monotonic_levels = monotonic_levels
        self.solution_set = None

    def graph_AMDP(self):
        if self.solution_set is not None:
            g = graphviz.Digraph('am_AMDP', format='png')

            for k, trajectory in enumerate(self.conflicting_trajectories):
                for l, triple in enumerate(trajectory):
                    state, action, reward, next_state = triple
                    # trajectories always start at base level (i.e., 0 here)
                    toggle_depth = self.depth
                    if l == 0:
                        toggle_depth = 1
                    for i in range(0, toggle_depth):
                        # imposing monotonicity on levels (i.e., can only move up in levels, not down)
                        j_start = 0
                        if self.monotonic_levels:
                            j_start = i
                        for j in range(j_start, self.depth):
                            if self.solution_set['traj_{}_triple_{}_({}^{},{},{}^{})'.format(k, l, state, i, action, next_state, j)] == 1:
                                g.node('{}^{}'.format(state, i), shape='box')
                                g.node('{}^{}'.format(next_state, j), shape='box')
                                g.edge('{}^{}'.format(state, i), '{}^{}'.format(next_state, j), label='a={}/r={}'.format(action, reward))
                                break
            g.render(filename="graphs/am_AMDP", format="png")

    def get_triggers(self):
        if self.solution_set is not None:
            triggers = dict()
            for k, trajectory in enumerate(self.conflicting_trajectories):
                for l, triple in enumerate(trajectory):
                    state, action, reward, next_state = triple
                    # trajectories always start at base level (i.e., 0 here)
                    toggle_depth = self.depth
                    if l == 0:
                        toggle_depth = 1
                    for i in range(0, toggle_depth):
                        # imposing monotonicity on levels (i.e., can only move up in levels, not down)
                        j_start = 0
                        if self.monotonic_levels:
                            j_start = i
                        for j in range(j_start, self.depth):
                            if self.solution_set['traj_{}_triple_{}_({}^{},{},{}^{})'.format(k, l, state, i, action, next_state, j)] == 1:
                                if i != j:
                                    if self.granularity == 'state':
                                        triggers['{},{}'.format(i, next_state)] = j
                                    elif self.granularity == 'triple':
                                        triggers['{}^{},{},{}'.format(state, i, action, next_state)] = j
                                break
        return triggers

    def solve(self, depth = 2, min_obj = 0):
        self.depth = depth
        while True:
            conflict_resolver = Model("Conflict Resolver")

            transition_ambiguity_dict, reward_ambiguity_dict = self._sum_to_one_constraint(conflict_resolver, depth)
            self._reward_ambiguity_constraint(reward_ambiguity_dict, conflict_resolver)
            level_change_indicators = self._transition_ambiguity_constraint(transition_ambiguity_dict, conflict_resolver)

            # OBJECTIVE
            z = conflict_resolver.addVar(name='objective', vtype=GRB.INTEGER)
            conflict_resolver.addConstr(z == quicksum(level_change_indicators), name='objective_constr')
            conflict_resolver.addConstr(z >= min_obj, name='objective_floor_constr')
            conflict_resolver.setObjective(z, GRB.MINIMIZE)
            print("\nCurrent Depth: {}, Current min obj: {}".format(depth, min_obj))
            conflict_resolver.optimize()
            if conflict_resolver.status == GRB.OPTIMAL:
                min_obj = conflict_resolver.objVal
                self.solution_set = dict()
                self.depth = depth
                for v in conflict_resolver.getVars():
                    self.solution_set[v.varName] = v.x
                break
            else:
                depth += 1
                min_obj = 1
        return depth, min_obj

    def _sum_to_one_constraint(self, conflict_resolver, depth):

        transition_ambiguity_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        reward_ambiguity_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        for k, trajectory in enumerate(self.conflicting_trajectories):
            prev_toggles_by_next_state_level = defaultdict(list)
            for l, triple in enumerate(trajectory):
                state, action, reward, next_state = triple

                # trajectories always start at base level (i.e., 0 here)
                toggle_depth = depth
                if l == 0:
                    toggle_depth = 1

                split_triples = []
                toggles_by_state_level = defaultdict(list)
                toggles_by_next_state_level = defaultdict(list)
                for i in range(0, toggle_depth):

                    # imposing monotonicity on levels (i.e., can only move up in levels, not down)
                    j_start = 0
                    if self.monotonic_levels:
                        j_start = i

                    for j in range(j_start, depth):
                        triple_toggle = conflict_resolver.addVar(name='traj_{}_triple_{}_({}^{},{},{}^{})'.format(k, l, state, i, action, next_state, j), vtype=GRB.BINARY)
                        # for continuity
                        toggles_by_state_level[i].append(triple_toggle)
                        toggles_by_next_state_level[j].append(triple_toggle)
                        # for sum to 1
                        split_triples.append(triple_toggle)
                        if self.granularity == 'state':
                            # for transition ambiguity
                            transition_ambiguity_dict['{}'.format(next_state)][i][j].append(triple_toggle)
                            # for reward ambiguity
                            reward_ambiguity_dict['{}'.format(next_state)][i][j][reward].append(triple_toggle)
                        elif self.granularity == 'triple':
                            # for transition ambiguity
                            transition_ambiguity_dict['{},{},{}'.format(state, action, next_state)][i][j].append(triple_toggle)
                            # for reward ambiguity
                            reward_ambiguity_dict['{},{},{}'.format(state, action, next_state)][i][j][reward].append(triple_toggle)
                ## TRAJECTORY CONTINUITY CONSTR
                for prev_next_state_level, next_state_level_toggles in prev_toggles_by_next_state_level.items():
                    state_level_toggles = toggles_by_state_level[prev_next_state_level]
                    conflict_resolver.addConstr(quicksum(next_state_level_toggles) == quicksum(state_level_toggles))
                prev_toggles_by_next_state_level = toggles_by_next_state_level

                ## SUM TO ONE CONSTR
                conflict_resolver.addConstr(quicksum(split_triples) == 1)

        return transition_ambiguity_dict, reward_ambiguity_dict

    def _reward_ambiguity_constraint(self, reward_ambiguity_dict, conflict_resolver):

        for toggle_type, i_dict in reward_ambiguity_dict.items():
            for i, j_dict in i_dict.items():
                for j, reward_dict in j_dict.items():
                    if len(reward_dict.keys()) == 1:
                        continue
                    toggle_type_indicators = []
                    for reward, toggles in reward_dict.items():
                        toggle_type_indicator = conflict_resolver.addVar(name='{},{},{},{}'.format(toggle_type, i, j, reward), vtype=GRB.BINARY)
                        # toggle type indicator is 1 if any of the toggles are in use, i.e., quicksum(toggles) > 1
                        conflict_resolver.addConstr(quicksum(toggles) >= 0.6 - 100000 * (1 - toggle_type_indicator))
                        conflict_resolver.addConstr(quicksum(toggles) <= 0.4 + 100000 * (toggle_type_indicator))
                        toggle_type_indicators.append(toggle_type_indicator)
                    ## REWARD AMBIGUITY CONSTR
                    conflict_resolver.addConstr(quicksum(toggle_type_indicators) <= 1)

    def _transition_ambiguity_constraint(self, transition_ambiguity_dict, conflict_resolver):
        level_change_indicators = []
        for toggle_type, i_dict in transition_ambiguity_dict.items():
            for i, j_dict in i_dict.items():
                toggle_type_indicators = []
                for j, toggles in j_dict.items():
                    toggle_type_indicator = conflict_resolver.addVar(name='{},{},{}'.format(toggle_type, i, j), vtype=GRB.BINARY)
                    # toggle type indicator is 1 if any of the toggles are in use, i.e., quicksum(toggles) > 1
                    conflict_resolver.addConstr(quicksum(toggles) >= 0.6 - 100000 * (1 - toggle_type_indicator))
                    conflict_resolver.addConstr(quicksum(toggles) <= 0.4 + 100000 * (toggle_type_indicator))
                    toggle_type_indicators.append(toggle_type_indicator)
                    if i != j:
                        # for objective
                        level_change_indicators.append(toggle_type_indicator)
                ## TRANSITION AMBIGUITY CONSTR
                conflict_resolver.addConstr(quicksum(toggle_type_indicators) <= 1)
        return level_change_indicators

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
    AM = AbstractionMachine(trajectories, granularity='triple')
    depth, min_obj = AM.solve()
    AM.graph_AMDP()
    triggers = AM.get_triggers()
    for trigger, level in triggers.items():
        print(trigger, level)
