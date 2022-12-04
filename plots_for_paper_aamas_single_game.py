from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ternary

# import matplotlib.style as style
# style.use('tableau-colorblind10')

import IterativePlayer
import games 
import plot_utils

plt.rcParams["figure.dpi"] = 150

# game_name = "RPS"
# game_name = "RPSLS"
# game_name = "Matching Pennies"
# game_name = "Biased RPS"
# game_name = "weakRPS"
# game_name = "RPS + safe R"
# game_name = "Random game"
# game_name = "RPS with mixed moves"
# game_name = "RPS Abstain"
# game_name = "Cyclic game"
game_name = "Cyclic game (n=7)"
# game_name = "Albert's RPS + safe R"
# game_name = "Identity (50)"
# game_name = "Transitive game (n=7)"

N_SAMPLES = 10_000
num_best_responses = 100
tiebreak_fn = np.random.choice

game = games.game_dict[game_name]
initial_strategy_p1 = IterativePlayer.one_hot(0, game.shape[0])
initial_strategy_p2 = IterativePlayer.one_hot(0, game.shape[1])

worst_case = {
    "FP" : np.zeros((N_SAMPLES, num_best_responses // 2)),
    "AFP" : np.zeros((N_SAMPLES, num_best_responses // 2))
}

for sample_idx in range(N_SAMPLES):
    is_last_run = sample_idx == N_SAMPLES-1
    
    play_fp = IterativePlayer.run_fp(
        game, num_best_responses, initial_strategy_p1, initial_strategy_p2, tiebreak_fn, solve_nash=is_last_run
    )
    play_afp = IterativePlayer.run_afp(game, num_best_responses // 2, initial_strategy_p1, initial_strategy_p2, tiebreak_fn)
    
    worst_case["FP"][sample_idx] = play_fp.worst_case_payoff[::2,0]
    worst_case["AFP"][sample_idx] = play_afp.worst_case_payoff[:,0]

#%% Plot

save_names = {
    "Cyclic game (n=7)" : "cyclic_game_7",
    "Transitive game (n=7)" : "transitive_game_7",
    "Cyclic game (n=20)" : "cyclic_game_20",
    "Transitive game (n=20)" : "transitive_game_20"
}
save_name = save_names.get(game_name, game_name)

fig, ax = plt.subplots()
save = True

x = range(0,num_best_responses,2)
scale = np.array(range(0,num_best_responses,2))[None,:]

plot_utils.plot_with_quantiles(ax, x, worst_case["AFP"], lw=2, c="C1", label="AFP")
plot_utils.plot_with_quantiles(ax, x, worst_case["FP"], lw=1.5, ls="--", c="C0", label="FP")

y_min = min(np.min(worst_case["FP"][:,1:]), np.min(worst_case["FP"][:,1:]))
ax.set_ylim([y_min-0.1,None])

ax.axhline(play_fp.value, ls=":", lw=1, label="Value of game", c="darkgrey")
ax.set_title(f"{game_name}")
ax.set_xlabel("Best responses calculated")
ax.set_ylabel("Worst case payoff")
ax.legend()

if save:
    plt.savefig(f"plots//generated//exploitability_{save_name}", bbox_inches='tight')
