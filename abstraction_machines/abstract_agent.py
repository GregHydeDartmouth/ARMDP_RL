import time
import json
import random
import graphviz
from abstraction_machines.abstraction_machine import AbstractionMachine
from collections import defaultdict

class AbstractAgent:
    def __init__(self, actions, granularity='state', monotonic_levels=False, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.2):
        self.AMDP = defaultdict(lambda: defaultdict(dict))
        self.actions = actions
        self.trajectory = []
        self.trajectories = []
        self.conflicting_trajectories = list()
        self.trajectory_mapping = defaultdict(set)
        self.conflict = None
        self.QSA = defaultdict(lambda: defaultdict(float))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.solve_time = 0

        # used to keep track of the level in each episode
        self.level = 0

        # parameters for abstraction
        self.granularity = granularity
        self.monotonic_levels = monotonic_levels
        self.triggers = dict()
        self.depth = 1
        self.min_obj = 0

    def save_triggers(self, name):
        with open('models/{}.json'.format(name), 'w') as f:
            json.dump(self.triggers, f)


    def step(self, state, action, reward, next_state, done):
        """
        Performs a single step of the agent in the environment and updates the trajectory.

        Args:
            state: The current state of the environment.
            action: The action taken by the agent.
            reward: The reward received from the environment.
            next_state: The next state of the environment after taking the action.

        Returns:
            None
        """

        self.trajectory.append((state, action, reward, next_state, done))
        state, action, next_state = self._abstract_triple(state, action, next_state)
        self._update_AMDP(state, action, reward, next_state, done)
        self._update_QSA(state, action, reward, next_state, done)
        return state, action, reward, next_state

    def _abstract_triple(self, state, action, next_state):
        if self.granularity == 'state':
            trigger = '{},{}'.format(self.level, next_state)
        elif self.granularity == 'triple':
            trigger = '{}^{},{},{}'.format(state, self.level, action, next_state)
        state = '{}^{}'.format(state, self.level)
        if trigger in self.triggers:
            self.level = self.triggers[trigger]
        next_state = '{}^{}'.format(next_state, self.level)
        return state, action, next_state

    def _update_AMDP(self, state, action, reward, next_state, done):
        if next_state not in self.AMDP[state][action]:
            self.AMDP[state][action][next_state] = {'reward':reward, 'done':done}
        else:
            found_reward = self.AMDP[state][action][next_state]['reward']
            if reward != found_reward:
                if self.conflict is None:
                    min_traj_idx = None
                    min_traj_len = None
                    for conflict_traj_idx in list(self.trajectory_mapping[(state,action,found_reward,next_state)]):
                        # conflict occurs in current traj
                        if conflict_traj_idx == len(self.trajectories):
                            continue
                        if min_traj_idx is None:
                            min_traj_idx = conflict_traj_idx
                            min_traj_len = len(self.trajectories[conflict_traj_idx])
                        else:
                            # need to figure out why this crashes occasionally. Shouldn't affect learning in general, but we might not always pick the shortest traj
                            try:
                                if len(self.trajectories[conflict_traj_idx]) < min_traj_len:
                                    min_traj_idx = conflict_traj_idx
                                    min_traj_len = len(self.trajectories[conflict_traj_idx])
                            except:
                                continue
                    if min_traj_idx is not None:
                        self.conflict = (min_traj_idx, len(self.trajectories))
        self.trajectory_mapping[(state,action,reward,next_state)].add(len(self.trajectories))

    def _update_QSA(self, state, action, reward, next_state, done):
        max_next_q_value = max(self.QSA[next_state].values()) if next_state in self.QSA else 0
        target_q_value = reward + self.discount_factor * max_next_q_value * (1-done)
        current_q_value = self.QSA[state][action]
        self.QSA[state][action] += self.learning_rate * (target_q_value - current_q_value)

    def choose_action(self, state):
        """
        Chooses an action to take in the given state using an epsilon-greedy policy.

        Args:
            state: The current state of the environment.

        Returns:
            The selected action.
        """
        state = '{}^{}'.format(state, self.level)
        if random.random() < self.exploration_rate:
            return random.choice(self.actions)
        else:
            for a in self.actions:
                if a not in self.QSA[state]:
                    self.QSA[state][a] = 0
            max_q_value = max(self.QSA[state].values())
            max_actions = [a for a in self.actions if self.QSA[state][a] == max_q_value]
            return random.choice(max_actions)

    def reset(self, update_conflicting_trajectories = True, make_graph_on_update=False, silent=False):
        """
        Resets the trajectory and saves a copy to the list of trajectories.

        Returns:
            None
        """
        if self.trajectory != []:
            self.trajectories.append(self.trajectory)
        self.trajectory = []
        self.level = 0
        solved_conflict = False
        if self.conflict is not None:
            self.conflicting_trajectories.append(self.trajectories[self.conflict[0]])
            self.conflicting_trajectories.append(self.trajectories[self.conflict[1]])
            self.AM = AbstractionMachine(self.conflicting_trajectories, granularity=self.granularity, monotonic_levels=self.monotonic_levels)
            if self.depth == 1:
                self.depth = 2
            t1 = time.time()
            self.depth, self.min_obj = self.AM.solve(depth = self.depth, min_obj = self.min_obj, silent=silent)
            t2 = time.time()
            self.solve_time += t2-t1
            self.triggers = self.AM.get_triggers()
            self.AMDP = defaultdict(lambda: defaultdict(dict))
            self.QSA = defaultdict(lambda: defaultdict(float))
            self.conflict = None
            self.trajectory_mapping = defaultdict(set)
            if update_conflicting_trajectories:
                self._update_conflicting_trajectories()
                if make_graph_on_update:
                    self.graph_AMDP()
                    self.graph_RM()
            solved_conflict = True
        return solved_conflict

    def _update_conflicting_trajectories(self):
        mapped_conflicting_trajectories = []
        for conflicting_trajectory in self.conflicting_trajectories:
            mapped_conflicting_trajectory = []
            self.reset()
            for conflicting_triple in conflicting_trajectory:
                state, action, reward, next_state, done = conflicting_triple
                state, action, reward, next_state = self.step(state, action, reward, next_state, done)
                mapped_conflicting_trajectory.append((state, action, reward, next_state, done))
            mapped_conflicting_trajectories.append(mapped_conflicting_trajectory)
        self.reset()
        for mapped_conflicting_trajectory in mapped_conflicting_trajectories:
            for mapped_conflicting_triple in reversed(mapped_conflicting_trajectory):
                state, action, reward, next_state, done = mapped_conflicting_triple
                self._update_QSA(state, action, reward, next_state, done)

    def graph_AMDP(self):
        g = graphviz.Digraph('aa_AMDP', format='png')
        edges = defaultdict(set)
        for state in self.AMDP:
            for action in self.AMDP[state]:
                for next_state in self.AMDP[state][action]:
                    reward = self.AMDP[state][action][next_state]['reward']
                    done = self.AMDP[state][action][next_state]['done']
                    qsa = self.QSA[state][action]
                    g.node(state, shape='circle')
                    g.node(next_state, shape='circle')
                    _label = 'a={}/r={}/qsa={}'.format(action, reward, qsa)
                    if _label not in edges[(state, next_state)]:
                        g.edge(state, next_state, label=_label)
                        edges[((state, next_state))].add(_label)
                    if done:
                        g.node('term', shape='box', color='red')
                        g.edge(next_state, 'term')
        g.render(filename="graphs/aa_AMDP", format="png")

    def graph_RM(self):
        g = graphviz.Digraph('am_RM', format='png')
        edges = defaultdict(set)
        for state in self.AMDP:
            for action in self.AMDP[state]:
                for next_state in self.AMDP[state][action]:
                    reward = self.AMDP[state][action][next_state]['reward']
                    done = self.AMDP[state][action][next_state]['done']
                    _state, rm_state = state.split('^')
                    _next_state, rm_next_state = next_state.split('^')

                    g.node(rm_state, shape='circle')
                    if self.granularity == 'state':
                        _label = '{}/{}'.format(_next_state, reward)
                    elif self.granularity == 'triple':
                        _label = '({},{},{})/{}'.format(_state, action, _next_state, reward)
                    if done:
                        node_2 = 'term'
                        g.node(node_2, shape='box', color='red')
                    else:
                        node_2 = rm_next_state
                        g.node(node_2, shape='circle')
                    if _label not in edges[(rm_state, node_2)]:
                        edges[(rm_state, node_2)].add(_label)
                    break
        for edge_nodes, edge_types in edges.items():
            if edge_nodes[1] != 'term':
                edge_label = None
                reward = None
                for edge_type in edge_types:
                    symbol, _reward = edge_type.split('/')
                    if edge_label is None:
                        edge_label = symbol
                        reward = _reward
                    else:
                        edge_label += 'V{}'.format(symbol)
                g.edge(edge_nodes[0], edge_nodes[1], label='{}/{}'.format(edge_label, reward))
            else:
                reward_labels = defaultdict(set)
                for edge_type in edge_types:
                    symbol, reward = edge_type.split('/')
                    reward_labels[reward].add(symbol)
                for reward, symbols in reward_labels.items():
                    edge_label = 'V'.join(symbols)
                    g.edge(edge_nodes[0], edge_nodes[1], label='{}/{}'.format(edge_label, reward))
        g.render(filename="graphs/aa_RM", format="png")

if __name__ == "__main__":
    actions = ['^', '>', '<', 'v']
    granularity = 'state'
    aa = AbstractAgent(actions, granularity=granularity, monotonic_levels=True)
    conflicts = 0
    trajectories = []
    t = [['1', '>', 0, '2', False],
        ['2', '^', 0, '6', False],
        ['6', '^', 0, '10', False],
        ['10', '^', 0, '14', False],
        ['14', '>', 0, '15', False],
        ['15', '>', 1, '16', True]]
    trajectories.append(t)
    t = [['1', '>', 0, '2', False],
        ['2', '^', 0, '6', False],
        ['6', '^', 0, '10', False],
        ['10', '>', 0, '11', False],
        ['11', '^', 0, '15', False],
        ['15', '>', 2, '16', True]]
    trajectories.append(t)
    for trajectory in trajectories:
        for triple in trajectory:
            state, action, reward, next_state, done = triple
            aa.step(state, action, reward, next_state, done)
        solved_conflict = aa.reset()
        if solved_conflict:
            conflicts += 1
    # if running granularity == 'state' this shouldn't trigger a resolve
    t = [['1', '>', 0, '2', False],
        ['2', '^', 0, '6', False],
        ['6', '^', 0, '10', False],
        ['10', 'v', 0, '11', False],
        ['11', '^', 0, '15', False],
        ['15', '>', 2, '16', True]]
    for triple in t:
        state, action, reward, next_state, done = triple
        aa.step(state, action, reward, next_state, done)
    solved_conflict = aa.reset()
    if solved_conflict:
            conflicts += 1
    if granularity == 'state':
        assert conflicts == 1, "state granularity not detected correctly"
    aa.graph_AMDP()
    aa.graph_RM()
    aa.save_triggers('test')
