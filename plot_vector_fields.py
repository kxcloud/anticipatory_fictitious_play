import numpy as np
import matplotlib.pyplot as plt
import ternary

import games 

"""
    NOTE: This code is hacky and hasn't been checked for correctness.
"""

action_names_dict = {
    "RPS" : ["Rock", "Paper", "Scissors"],
    "RPS Abstain" : ["Rock", "Paper", "Scissors", "Abstain"],
    "Biased RPS" : ["Rock", "Paper", "Scissors"],
    "weakRPS" : ["Rock", "Paper", "Scissors"],
    "Matching Pennies Abstain" : ["Heads", "Tails", "Abstain"],
    "Random game 1" : range(3),
    "Random game 2" : range(3),
    "Transitive game" : range(3)
}

def plot_vector_field_on_simplex(tax, game, plot_afp, step_size_alg=None, step_size_plot=0.06, density=12):
    assert np.equal(game,-game.T).all(), "game must be symmetric"
    assert plot_afp + (step_size_alg is None) == 1, "Need step size if and only if plotting AFP"
    
    points = []
    for i in range(density+1):
        for j in range(i+1):
            x1 = 1 - i/density
            x2 = 0 if i == 0 else j/i * (1 - x1)
            x3 = 1-x1-x2
            point = np.array([x1, x2, x3])
            assert np.abs(point.sum()-1) < 1e-8, f"{point} must sum to 1"
            assert np.min(point >= -1e-8), f"{point} must be positive"
            points.append(point)
    prob_grid = np.array(points)
    best_response = np.argmin(prob_grid @ game, axis=1)
    
    for point, br in zip(prob_grid, best_response):
        if plot_afp:
            fp_point = point*(1-step_size_alg)
            fp_point[br] += step_size_alg
            br_br = np.argmin(fp_point)
            afp_point = point*(1-step_size_plot)
            afp_point[br_br] += step_size_plot
            
            tax.plot([point, afp_point], c="black")
        else:
            fp_point = point*(1-step_size_plot)
            fp_point[br] += step_size_plot
            tax.plot([point, fp_point], c="black")
    
    tax.scatter(prob_grid, s=10, zorder=-10, c="black")
    return tax

fig, axes = plt.subplots(ncols=4, figsize=(15,4))
taxes = [ternary.TernaryAxesSubplot(ax=ax) for ax in axes]

plot_vector_field_on_simplex(taxes[0], games.game_dict["RPS"], plot_afp=False)
axes[0].set_aspect('equal', adjustable='box')
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[0].set_title("FP")

for ax, tax, timestep in zip(axes[1:], taxes[1:], [2, 4, 100]):
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    
    step_size = 1/timestep
    plot_vector_field_on_simplex(tax, games.game_dict["weakRPS"], step_size_alg=step_size, plot_afp=True)
    ax.set_title(f"AFP (t={timestep})")

plt.show()