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
num_game_samples = 5000
num_best_responses = 15
tiebreak_fn = np.random.choice

shared_args = {
    "initial_strategy_p1" : IterativePlayer.one_hot(0, game_shape[0]),
    "initial_strategy_p2" : IterativePlayer.one_hot(0, game_shape[1]),
    "tiebreak_fn" : tiebreak_fn
}

alg_dict = {
    "FP" : partial(IterativePlayer.run_fp, **shared_args), 
    "FP " : partial(IterativePlayer.run_fp, **shared_args), 
    "AFP" : partial(IterativePlayer.run_afp_parity, **shared_args),
    "AFP (1 initial FP step)" : partial(IterativePlayer.run_afp_parity, **shared_args, num_initial_fp_steps=1),
    "AFP (2 initial FP steps)" : partial(IterativePlayer.run_afp_parity, **shared_args, num_initial_fp_steps=2),
    "AFP (3 initial FP steps)" : partial(IterativePlayer.run_afp_parity, **shared_args, num_initial_fp_steps=3)
}

alg_names = list(alg_dict.keys())
alg_indexes = {name : idx for idx, name in enumerate(alg_names)}

avg_worst_case_payoffs = {    
    name: np.zeros((num_game_samples, num_best_responses, 2)) for name in alg_names
}

pct_of_time_better_than_first_alg = {
    name: np.zeros((num_game_samples, num_best_responses)) for name in alg_names[1:]
}

for game_sample_idx in range(num_game_samples):
    game = np.random.normal(size=game_shape)
    
    for k, (alg_name, alg) in enumerate(alg_dict.items()):
        play = alg(game, num_best_responses)
        performance = play.worst_case_payoff
        
        avg_worst_case_payoffs[alg_name][game_sample_idx] = performance
            
        if k == 0:
            first_alg_performance = performance[:,0]
        
        if k > 0:
            is_better = (np.sign(performance[:,0] - first_alg_performance) + 1)/2
            pct_of_time_better_than_first_alg[alg_name][game_sample_idx] = is_better
 
        
#%%
save = True
x_vals = list(range(num_best_responses))

fig, ax = plt.subplots()

for idx, name in enumerate(alg_names):
    plot_utils.plot_with_quantiles(ax, x_vals, avg_worst_case_payoffs[name][:,:,0], c=f"C{idx}", label=name)
# plot_utils.plot_with_quantiles(ax, x_vals, avg_worst_case_payoffs["FP"][:,1:,0], c="C0", ls="--", label="FP")
ax.set_ylabel("Mean worst-case payoff")
ax.set_xlabel("Best responses calculated")
ax.set_title("Performance on random 30x30 matrices")
ax.legend()

fig, ax = plt.subplots()
# ax.axhline(1, lw=1, ls=":", c="black", alpha=0.7)

names_to_plot = ["FP ", "AFP", "AFP (1 initial FP step)", "AFP (2 initial FP steps)", "AFP (3 initial FP steps)"]
linestyles = ["-","--",":","-.","-"]

for name, ls in zip(names_to_plot, linestyles):
    color = f"C{alg_indexes[name]}"
    plot_utils.plot_with_agresti_coull(ax, x_vals[::2], pct_of_time_better_than_first_alg[name][:,::2], c=color, lw=2, label=name, ls=ls)


# means = np.mean(pct_of_time_better_than_first_alg["AFP"][:,1:], axis=0)

# for threshold in [0.5, 0.9]:
#     # Linearly interpolate, assumes monotonic data
#     idx_before = np.max(np.argwhere(means < threshold))
#     idx_after = idx_before + 1
#     x_before, x_after = x_vals[idx_before], x_vals[idx_after]
#     y_before, y_after = means[idx_before], means[idx_after]
#     x_interpolated = x_before + (x_after - x_before) * (threshold-y_before) / (y_after - y_before)
#     plot_utils.indicate_point(ax, x_interpolated, threshold, color="black", lw=1.25, ls=":", alpha=1)

# max_mean = np.max(means)
# if max_mean < 1:
#     print("Warning: maximum proportion {max_mean} < 1, plot was intended to show value at 1.")

# idx_first_1 = np.min(np.argwhere(means==max_mean))
# plot_utils.indicate_point(ax, x_vals[idx_first_1], 1, color="black", lw=1.25, ls=":", alpha=1)

ax.set_ylabel("Proportion")
ax.set_xlabel("Best responses calculated")
shape_str = f"{game_shape[0]}x{game_shape[1]}"
ax.set_title(f"Proportion of the time AFP is better than FP\non random {shape_str} matrices")
ax.legend()

if save:
    plt.savefig("plots//generated//proportion_afp_with_init_better", bbox_inches='tight')
plt.show()
