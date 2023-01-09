from abstraction_machines.abstract_machine import AbstractionMachine
from beachworld import Beachworld
from collections import deque

BW = Beachworld()
gamma = 0.9
eps = 0.1
max_traj_length = 10
action_set=['^','v','<','>','o']
AM = AbstractionMachine(action_set=action_set, num_models=1, verbose=True)
rewards = deque(maxlen=100)

for i in range(0, 10000):
    state = BW.reset()
    AM.reset()
    done = False

    _conflict = False
    _new_q_val = False
    trajectory = []
    trajectory_reward = 0
    transitions = 0

    while not done:
        action = AM.get_action(state, eps=eps)
        reward, next_state, done = BW.step(action)
        trajectory_reward += reward
        conflict, new_q_val = AM.step(state, action, reward, next_state)
        if conflict:
            _conflict=True
        if new_q_val:
            _new_q_val=True
        if transitions == max_traj_length:
            done = True
        trajectory.append([state, action, reward, next_state])
        state = next_state
        transitions += 1
    AM.update_abstractions(trajectory, resolve_conflict=_conflict, resolve_q_vals=_new_q_val, make_graph=True)
    rewards.append(trajectory_reward)
    avg_reward = sum(rewards)/len(rewards)
    print('\ravg reward: {}, episode_num: {}'.format(avg_reward, i), end="")
#AM._build_abstract_MDP_graph()
AM._solve_abstract_MDP()
AM.save_abstraction_model("test.json")
AM.load_abstraction_model("test.json")
AM._build_abstract_MDP_graph()
