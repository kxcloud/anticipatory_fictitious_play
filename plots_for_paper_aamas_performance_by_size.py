import time

import numpy as np
import matplotlib.pyplot as plt

import IterativePlayer
import plot_utils

plt.rcParams["figure.dpi"] = 150
    
#%% Performance by size

start_time = time.time()
num_reps = 2500
t_max = 50
tiebreak_fn = np.random.choice

n_vals = range(2,201)

avg_worst_case_payoff = {
   "FP" : np.zeros((num_reps, len(n_vals))),
   "AFP" : np.zeros((num_reps, len(n_vals)))
}

results = []
for n_idx, n in enumerate(n_vals):
    print(f"n={n}...")
    avg_worst_case_payoff_fp = 0 
    avg_worst_case_payoff_afp = 0 
    
    for rep_idx in range(num_reps):
        game = np.random.normal(size=(n,n))
        initial_strategy_p1 = IterativePlayer.one_hot(0, game.shape[0])
        initial_strategy_p2 = IterativePlayer.one_hot(0, game.shape[1])
        
        play_fp = IterativePlayer.run_fp(game, 2*t_max, initial_strategy_p1, initial_strategy_p2, tiebreak_fn)
        avg_worst_case_payoff["FP"][rep_idx][n_idx] = play_fp.worst_case_payoff[-1, 0]
        
        play_afp = IterativePlayer.run_afp(game, t_max, initial_strategy_p1, initial_strategy_p2, tiebreak_fn)
        avg_worst_case_payoff["AFP"][rep_idx][n_idx] = play_afp.worst_case_payoff[-1, 0]
        
duration = (time.time() - start_time)/60
print(f"{duration:0.2} min for {num_reps} replicates")
#%%
save = True
fig, ax = plt.subplots()
q=[0.1,0.9]
plot_utils.plot_with_quantiles(ax, n_vals, avg_worst_case_payoff["FP"], 
                               q=q, c="C0", ls="--", lw=1.5, label="FP")
plot_utils.plot_with_quantiles(ax, n_vals, avg_worst_case_payoff["AFP"],
                               q=q, c="C1", lw=2, label="AFP")
ax.set_ylim([-0.55, 0.3])
ax.set_xticks(sorted(list(ax.get_xticks())+[2]))
ax.set_xlim([min(n_vals), max(n_vals)])
# ax.plot(results[:,0], results[:,1],label="FP")
# ax.plot(results[:,0], results[:,2],label="AFP")
ax.set_ylabel("Worst case payoff")
ax.set_xlabel("Matrix width and height")

ax.set_title(f"Performance on random matrices at {2*t_max}th response")
ax.legend()

if save:
    plt.savefig(f"plots//generated//performance_by_size", bbox_inches='tight')