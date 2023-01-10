from functools import partial
import os
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import ternary
from scipy.stats import norm

import IterativePlayer
import games 
import plot_utils
import lp_solver

code_path = os.path.dirname(os.path.realpath(__file__))
project_path = code_path
data_path = os.path.join(project_path,"data")
plots_out_path = os.path.join(project_path, "plots", "rl_plots")

plt.rcParams["figure.dpi"] = 150
import matplotlib.style as style
style.use('default')

ALPHA = 0.1

def mean_with_ci(data, alpha=ALPHA):
    mean = np.mean(data)
    ci_width = norm.ppf(1-alpha/2) * np.std(data) / np.sqrt(len(data))
    return mean, ci_width

def get_opponents(index, alg_type):
    if alg_type == "fp":
        return range(index)
    
    if alg_type == "afp":
        eligible = list(range(0, index, 2))
        if index % 2 == 0:
            eligible.append(index - 1)
        return eligible

    assert False

def get_payoff_matrices_by_type(filepath):
    df = pd.read_csv(filepath)

    for agent_idx in ["1","2"]:
        df[f"type_{agent_idx}"] = df[f"policy_{agent_idx}"].apply(
            lambda x: 'afp' if 'afp' in x else 'fp' if 'fp' in x else 'ERROR'
        )
    
    df_fp_v_afp = df[(df.type_1 == 'fp') & (df.type_2 == 'afp')].copy(deep=True)
    df_fp_v_fp = df[(df.type_1 == 'fp') & (df.type_2 == 'fp')].copy(deep=True)
    df_afp_v_afp = df[(df.type_1 == 'afp') & (df.type_2 == 'afp')].copy(deep=True)
    
    dfs = {"fp_v_afp" : df_fp_v_afp, "fp_v_fp": df_fp_v_fp, "afp_v_afp" : df_afp_v_afp }
    
    for sub_df in [df_fp_v_afp, df_fp_v_fp, df_afp_v_afp]:
        exp_ids = list(sub_df.experiment_index.unique())
        id_remapping = {exp_id: i for i, exp_id in enumerate(exp_ids)}
        
        sub_df["new_id"] = sub_df["experiment_index"].apply(lambda x: id_remapping[x])
    
    # Construct payoff matrices
    population_size = df[INDEX_VAR+"1"].max()+1
    assert population_size == df[INDEX_VAR+"2"].max()+1, "Expect same number of agent indices in both experiment types"
    shape = (len(df_fp_v_afp.experiment_index.unique()), population_size, population_size)
    
    payoff_matrices_by_type = {key: np.zeros(shape) for key in dfs.keys()}
    
    for key, df in dfs.items():
        for _, row in df.iterrows():
            self_play = key.split("_v_")[0] == key.split("_v_")[1] 
            if not self_play:
                payoff_matrices_by_type[key][row.new_id, row[INDEX_VAR+"1"], row[INDEX_VAR+"2"]] += row['avg(reward_1)']
                
            if self_play: # Use symmetry of game
                payoff_matrices_by_type[key][row.new_id, row[INDEX_VAR+"1"], row[INDEX_VAR+"2"]] += row['avg(reward_1)']/2
                payoff_matrices_by_type[key][row.new_id, row[INDEX_VAR+"2"], row[INDEX_VAR+"1"]] -= row['avg(reward_1)']/2
    
    return payoff_matrices_by_type

INDEX_VAR = "neupl_idx_" 

read_path = "data\\"
save_path = "G:\\System\\Documents\\ACADEMIC\\mmfp\\\plots\\rl_plots"

SAVE_FIGURES = True

# %% Load and clean data

filenames = glob.glob(os.path.join(data_path,"*0.csv"))
# filenames = glob.glob(os.path.join(data_path,"footsies-p11-4850.csv"))

for filepath in filenames:
    filename = filepath.split("\\")[-1].split(".")[0]
    print(filename)
    payoff_matrices_by_type = get_payoff_matrices_by_type(filepath)
    population_size = payoff_matrices_by_type["fp_v_fp"].shape[-1]
    # fig, axes = plt.subplots(nrows=5, figsize=(3,10))
    # for newid in range(5):
    #     axes[newid].imshow(payoff_matrices[newid], cmap="RdBu")
        
    #%% Results
    payoff_matrices = payoff_matrices_by_type['fp_v_afp']
    
    # Head-to-head
    mean, ci_width = mean_with_ci(payoff_matrices[:, :, get_opponents(8,'afp')])
    head_to_head_str = f"Mean head-to-head (FP vs AFP): {mean:0.2f} ± {ci_width:0.2f}"
    print(head_to_head_str)
    
    # Relative population performance
    values = []
    for payoff_matrix in payoff_matrices:
        p2_probs, info = lp_solver.solve_game_half(payoff_matrix)
        p1_probs, _ = lp_solver.solve_game_half(-payoff_matrix.T)
        value = info["res"]["fun"]
        values.append(value)
    
        print(f"p1: {np.round(p1_probs,2)}")
        print(f"p2: {np.round(p2_probs,2)}")
        print(f"value: {value:0.2f}")
                
    mean, ci_width = mean_with_ci(values)
    rpp_string = f"Mean relative population performance (FP vs AFP): {mean:0.2f} ± {ci_width:0.2f}"
    print(rpp_string)
    
    #%% Empirical game
    fig, ax = plt.subplots()
    pc = ax.pcolormesh(payoff_matrices.mean(axis=0), 
                       norm=colors.CenteredNorm(), cmap="RdBu")
    fig.colorbar(pc, ax=ax)
    
    ax.set_ylabel("FP index")
    ax.set_xlabel("AFP index")
    
    titles = ["Average empirical game for FP vs. AFP (FP payoffs)", head_to_head_str, rpp_string]
    ax.set_title("\n".join(titles))
    
    freq_to_tick = 1 if population_size <= 12 else 5
    ax.set_xticks(np.arange(0, population_size, freq_to_tick)+0.5)
    ax.set_yticks(np.arange(0, population_size, freq_to_tick)+0.5)
    ax.set_xticklabels(np.arange(0, population_size, freq_to_tick))
    ax.set_yticklabels(np.arange(0, population_size, freq_to_tick))
    
    if SAVE_FIGURES:
        plt.tight_layout()
        plt.savefig(os.path.join(plots_out_path, f"{filename}_empirical_game.png"))
    
    #%%
    fig, axes = plt.subplots(ncols=2, figsize=(8,3))
    
    for ax, alg in zip(axes, ["fp", "afp"]):
        payoff_matrix_stack = payoff_matrices_by_type[f'{alg}_v_{alg}']
        pc = ax.pcolormesh(payoff_matrix_stack.mean(axis=0),
                            norm=colors.CenteredNorm(), cmap="RdBu")
        fig.colorbar(pc, ax=ax)
        
        ax.set_ylabel("Agent index")
        ax.set_xlabel("Agent index")
        
        titles = [f"Empirical game for {alg} vs. {alg}: {filename}"]
        ax.set_title("\n".join(titles))
        
        freq_to_tick = 1 if population_size <= 12 else 5
        ax.set_xticks(np.arange(0, population_size, freq_to_tick)+0.5)
        ax.set_yticks(np.arange(0, population_size, freq_to_tick)+0.5)
        ax.set_xticklabels(np.arange(0, population_size, freq_to_tick))
        ax.set_yticklabels(np.arange(0, population_size, freq_to_tick))
    
    #%% Exploitability
        
    all_exploitability = {}
    for alg_type in ["fp", "afp"]:
        payoff_matrices_alg = payoff_matrices_by_type[f"{alg_type}_v_{alg_type}"]
        num_experiments = payoff_matrices_alg.shape[0]
        all_exploitability[alg_type] = np.zeros((num_experiments, population_size-1))
            
        for experiment_idx in range(num_experiments):
            for agent_idx in range(population_size-1):
                exploit_idx = agent_idx+1
                exploit = np.mean(payoff_matrices_alg[experiment_idx, exploit_idx, get_opponents(exploit_idx,alg_type)])
                all_exploitability[alg_type][experiment_idx, agent_idx] = exploit
    
    exploitability = {}
    exploit_ci_width = {}
    for alg_type in ["fp", "afp"]:
        exploitability[alg_type] = np.mean(all_exploitability[alg_type], axis=0)
        exploit_ci_width[alg_type] = norm.ppf(1-ALPHA/2)*np.std(all_exploitability[alg_type], axis=0) / np.sqrt(len(exploitability[alg_type]))
        
    #%% Plot exploitability
    fig, ax = plt.subplots()
    
    plot_utils.plot_with_confidence(ax, x=range(population_size-1), data=all_exploitability["afp"], c="C1", label="AFP", lw=2, alpha=ALPHA)
    plot_utils.plot_with_confidence(ax, x=range(population_size-1), data=all_exploitability["fp"], c="C0", label="FP", ls="--", lw=1.6, alpha=ALPHA)
    # plot_utils.plot_with_confidence(ax, x=range(0,population_size-1,2), data=all_exploitability["afp"][:,::2], c="C1", label="AFP", lw=1.5)
    ax.set_xlabel("Agent index")
    ax.set_xticks(range(population_size-1))
    ax.set_ylabel("Worst case payoff")
    ax.set_title(f"Exploitability: {filename}")
    ax.legend()
    
    if SAVE_FIGURES:
        plt.tight_layout()
        plt.savefig(os.path.join(plots_out_path, f"{filename}_exploitability.png"))
