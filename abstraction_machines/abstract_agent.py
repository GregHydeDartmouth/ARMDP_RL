import random
from abstraction_machines.abstraction_machine import AbstractionMachine
from collections import defaultdict

class AbstractAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.AMDP = defaultdict(lambda: defaultdict(dict))
        self.actions = actions
        self.trajectory = []
        self.trajectories = []
        self.conflicting_trajectories = list()
        self.trajectory_mapping = defaultdict(set)
        self.conflict = None
        self.T = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.QSA = defaultdict(lambda: defaultdict(float))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

        # parameters for abstraction
        self.triggers = dict()
        self.depth = 2
        self.min_obj = 0

    def step(self, state, action, reward, next_state):
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
        self._update_AMDP(state, action, reward, next_state)
        self._update_QSA(state, action, reward, next_state)
        self.trajectory.append((state, action, reward, next_state))

    def _update_AMDP(self, state, action, reward, next_state):
        if next_state not in self.AMDP[state][action]:
            self.AMDP[state][action][next_state] = {'reward':reward}
        else:
            found_reward = self.AMDP[state][action][next_state]['reward']
            if reward != found_reward:
                if self.conflict is None:
                    self.conflict = (list(self.trajectory_mapping[(state,action,found_reward,next_state)])[-1], len(self.trajectories))
        self.trajectory_mapping[(state,action,reward,next_state)].add(len(self.trajectories))
    
    def _update_QSA(self, state, action, reward, next_state):
        max_next_q_value = max(self.QSA[next_state].values()) if next_state in self.QSA else 0
        target_q_value = reward + self.discount_factor * max_next_q_value
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
        if random.random() < self.exploration_rate or len(self.AMDP[state]) == 0:
            return random.choice(self.actions)
        else:
            for a in self.actions:
                if a not in self.QSA[state]:
                    self.QSA[state][a] = 0
            max_q_value = max(self.QSA[state].values())
            max_actions = [a for a in self.actions if self.QSA[state][a] == max_q_value]
            return random.choice(max_actions)

    def reset(self):
        """
        Resets the trajectory and saves a copy to the list of trajectories.

        Returns:
            None
        """
        if self.trajectory != []:
            self.trajectories.append(self.trajectory)
        self.trajectory = []
        if self.conflict is not None:
            self.conflicting_trajectories.append(self.trajectories[self.conflict[0]])
            self.conflicting_trajectories.append(self.trajectories[self.conflict[1]])
            self.AM = AbstractionMachine(self.conflicting_trajectories)
            self.depth, self.min_obj = self.AM.solve(depth = self.depth, min_obj = self.min_obj)
            self.triggers = self.AM.get_triggers()
            self.conflict = None
