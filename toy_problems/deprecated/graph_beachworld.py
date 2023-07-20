import json
import numpy as np
import matplotlib.pyplot as plt

with open('beachworld.json', 'r') as f:
    x = json.load(f)

run_is = x['is']
# pick any
_is = run_is[0]

run_reward_avgs = x['run_reward_avgs']
run_rewards_array = np.array(run_reward_avgs)
run_rewards_array_max = np.max(run_rewards_array, axis = 0)
run_rewards_array_min = np.min(run_rewards_array, axis = 0)
run_rewards_array_avg = np.mean(run_rewards_array, axis = 0)

run_depths = x['depths']
run_depths_array = np.array(run_depths)
run_depths_max = np.max(run_depths_array, axis = 0)
run_depths_min = np.min(run_depths_array, axis = 0)
run_depths_avg = np.mean(run_depths_array, axis = 0)

run_solve_times = x['run_solve_times']
run_solve_times_array = np.array(run_solve_times)
run_solve_times_array_max = np.max(run_solve_times_array, axis = 0)
run_solve_times_array_min = np.min(run_solve_times_array, axis = 0)
run_solve_times_array_avg = np.mean(run_solve_times_array, axis = 0)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6,12))
#fig, ax1 = plt.subplots()
ax1.plot(_is, run_rewards_array_avg, color='blue', label='AMDP')
ax1.set_ylim([0,max(run_rewards_array_max)+1])
ax1.fill_between(_is, run_rewards_array_max, run_rewards_array_min, alpha=0.2)
ax1.axhline(y=5.5, color='red', label='optimal')
ax1.set_ylabel('trajectory reward')
ax1.legend()

ax2.plot(_is, run_depths_avg, color='blue', label='avg')
ax2.plot(_is, run_depths_max, color='orange', label='max')
ax2.plot(_is, run_depths_min, color='green', label='min')
ax2.set_ylabel('model depth')
ax2.legend()

ax3.plot(_is, run_solve_times_array_avg, color='blue', label='avg')
ax3.plot(_is, run_solve_times_array_max, color='orange', label='max')
ax3.plot(_is, run_solve_times_array_min, color='green', label='min')
ax3.set_ylabel('cumulative solve time (s)')
ax3.set_xlabel('trajectory num')
ax3.legend()

plt.savefig('test_figure.png')
plt.close()
