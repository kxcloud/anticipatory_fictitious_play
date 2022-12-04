from functools import partial

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import ternary

import IterativePlayer
import games 
import plot_utils

plt.rcParams["figure.dpi"] = 150
import matplotlib.style as style
style.use('default')
    
#%% Average performance over fixed size
game_shape = (20,20)

#
# SETTINGS #
#
num_game_samples = 3000
num_best_responses = 200
tiebreak_fn = np.random.choice

alg_names = ["FP", "AFP"] #, "AFP (1 initial FP step)", "AFP (2 initial FP steps)"]
shared_args = {
    "initial_strategy_p1" : IterativePlayer.one_hot(0, game_shape[0]),
    "initial_strategy_p2" : IterativePlayer.one_hot(0, game_shape[1]),
    "tiebreak_fn" : tiebreak_fn
}
alg_list = [
    partial(IterativePlayer.run_fp, **shared_args), 
    partial(IterativePlayer.run_afp, **shared_args),
    # partial(IterativePlayer.run_afp_parity, **shared_args, num_initial_fp_steps=1),
    # partial(IterativePlayer.run_afp_parity, **shared_args, num_initial_fp_steps=2),
]
alg_br_per_timestep = [1, 2] #, 1, 1]

num_comparison_steps = num_best_responses // 2

avg_worst_case_payoffs = {    
    name: np.zeros((num_game_samples, num_comparison_steps, 2)) for name in alg_names
}

pct_of_time_better_than_first_alg = {
    name: np.zeros((num_game_samples, num_comparison_steps)) for name in alg_names[1:]
}

steps = { "FP" : num_best_responses, "AFP" : num_comparison_steps, 
         "AFP (1 initial FP step)": num_best_responses,  "AFP (2 initial FP steps)": num_best_responses}

for game_sample_idx in range(num_game_samples):
    game = np.random.normal(size=game_shape)
    
    for k, (alg_name, alg, br_per_step) in enumerate(zip(alg_names, alg_list, alg_br_per_timestep)):
        play = alg(game, steps[alg_name])
        performance = play.worst_case_payoff[::3-br_per_step,:]
        
        avg_worst_case_payoffs[alg_name][game_sample_idx] = performance
            
        if k == 0:
            first_alg_performance = performance[:,0]
        
        if k > 0:
            is_better = (np.sign(performance[:,0] - first_alg_performance) + 1)/2
            pct_of_time_better_than_first_alg[alg_name][game_sample_idx] = is_better
            # pct_of_time_better_than_first_alg[alg_name][game_sample_idx] = first_alg_performance < performance[:,0]
 
        
#%%
save = True
x_vals = list(range(0,num_best_responses,2))

# fig, ax = plt.subplots()

# plot_utils.plot_with_quantiles(ax, x_vals, avg_worst_case_payoffs["AFP"][:,:,0], c="C1", ls="--", label="AFP")
# # plot_utils.plot_with_quantiles(ax, x_vals, avg_worst_case_payoffs["AFP (1 initial FP step)"][:,:,0], c="black", ls=":", label="AFP (1 initial FP step)")
# plot_utils.plot_with_quantiles(ax, x_vals, avg_worst_case_payoffs["FP"][:,:,0], c="C0", ls="-.", label="FP")
# ax.set_ylabel("Mean worst-case payoff")
# ax.set_xlabel("Best responses calculated")
# ax.set_title("Performance on random 30x30 matrices")
# ax.legend()

fig, ax = plt.subplots()
# ax.axhline(1, lw=1, ls=":", c="black", alpha=0.7)

plot_utils.plot_with_agresti_coull(ax, x_vals, pct_of_time_better_than_first_alg["AFP"][:,:], c="C2", lw=2, label="AFP")
# plot_utils.plot_with_agresti_coull(ax, x_vals, 
#                                    pct_of_time_better_than_first_alg["AFP (1 initial FP step)"][:,:],
#                                    c="C3", lw=2, ls=":", label="AFP (1 initial FP step)")
# plot_utils.plot_with_agresti_coull(ax, x_vals, 
#                                    pct_of_time_better_than_first_alg["AFP (2 initial FP steps)"][:,:],
#                                    c="C4", lw=2, ls="-.", label="AFP (2 initial FP steps)")
ax.legend()

means = np.mean(pct_of_time_better_than_first_alg["AFP"][:,1:], axis=0)

for threshold in [0.5, 0.9]:
    # Linearly interpolate, assumes monotonic data
    idx_before = np.max(np.argwhere(means < threshold))
    idx_after = idx_before + 1
    x_before, x_after = x_vals[idx_before], x_vals[idx_after]
    y_before, y_after = means[idx_before], means[idx_after]
    x_interpolated = x_before + (x_after - x_before) * (threshold-y_before) / (y_after - y_before)
    plot_utils.indicate_point(ax, x_interpolated, threshold, color="black", lw=1.25, ls=":", alpha=1)

idx_first_1 = np.min(np.argwhere(means==1))
plot_utils.indicate_point(ax, x_vals[idx_first_1], 1, color="black", lw=1.25, ls=":", alpha=1)

ax.set_ylabel("Proportion")
ax.set_xlabel("Best responses calculated")
ax.set_title("Proportion of the time AFP is better than FP\non random 30x30 matrices")

if save:
    plt.savefig("plots//generated//proportion_afp_better_fp_30x30", bbox_inches='tight')
plt.show()