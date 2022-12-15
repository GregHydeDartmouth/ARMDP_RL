import gym, envs
from abstract_machine import AbstractionMachine

env = gym.make('Office-v0')
optimal_trajectories = env.get_policy_data(num_episodes=1000, epsilon=0.1, gamma=0.9)
noise_trajectories = env.get_policy_data(num_episodes=1000, epsilon=1, gamma=0.9)
trajectories = optimal_trajectories + noise_trajectories
AM = AbstractionMachine(trajectories)
AM.resolve_reward_conflicts(write_file=True, make_graph=True)



