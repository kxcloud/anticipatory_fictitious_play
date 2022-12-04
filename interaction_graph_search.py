from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.special import softmax

import plot_metagraphs

"""
    Search over interaction graphs which implement Perturbed Fictitous Play 
    to find ones that are better than FP with high probability.
"""

def get_argmaxes(values):
    argmaxes = [None]
    max_val = -np.inf
    for idx, val in enumerate(values):
        if val > max_val:
            argmaxes = [idx]
            max_val = val
        elif val == max_val:
            argmaxes.append(idx)
    return argmaxes, max_val


def get_interaction_graph(n, num_fp_indices=None, fp_indices=None):
    assert (num_fp_indices is None) + (fp_indices is None) == 1
    assert n > 5
    
    if fp_indices is None:
        fp_indices = list(np.random.choice(n-1, size=num_fp_indices-1, replace=False))
        fp_indices.append(n-1) # Ensure final index is FP step
        fp_indices = sorted(fp_indices)

    non_fp_indices = [idx for idx in range(n) if idx not in fp_indices]     
    
    interaction_graph = np.zeros((n,n))    
    for learner_idx in fp_indices:
        for opp_idx in fp_indices:
            if opp_idx == learner_idx:
                break
            interaction_graph[learner_idx, opp_idx] = 1
        
        if learner_idx > 0:
            interaction_graph[learner_idx, learner_idx-1] = 1 # Include "perturbation" step
            interaction_graph[learner_idx,:] /= interaction_graph[learner_idx,:].sum()
            
    for learner_idx in non_fp_indices:
        if learner_idx > 0:
            # interaction_graph[learner_idx, :learner_idx] = np.random.dirichlet([1]*(learner_idx))
            interaction_graph[learner_idx, :learner_idx-1] = np.random.binomial(n=1, p=0.5, size=learner_idx-1)
            interaction_graph[learner_idx, learner_idx-1] = 1
            interaction_graph[learner_idx,:] /= interaction_graph[learner_idx,:].sum()
    
    return interaction_graph, fp_indices


def evaluate_interaction_graph(igraph, game):
    assert type(igraph) is np.ndarray
    assert type(game) is np.ndarray
    
    strategies_p0 = [0]
    strategies_p1 = [0]
    
    exploitabilities = []
    
    for t in range(1,len(igraph)):
        # Does this go far enough
        p0_payoffs = game[:,strategies_p1] @ igraph[t,:t]
        p1_payoffs = -game[strategies_p0].T @ igraph[t,:t]
        p0_argmaxes, p0_max = get_argmaxes(p0_payoffs)
        p1_argmaxes, p1_max = get_argmaxes(p1_payoffs)
        
        p0_strat = np.random.choice(p0_argmaxes)
        p1_strat = np.random.choice(p1_argmaxes)
        
        strategies_p0.append(p0_strat)
        strategies_p1.append(p1_strat)
        
        exploitabilities.append(p0_max)
    return exploitabilities

    
def evaluate_interaction_graph_many(igraph, games_list):
    final_exploit = np.zeros(len(games_list))
    for game_idx, game in enumerate(games_list):
        final_exploit[game_idx] = evaluate_interaction_graph(igraph, game)[-1]
    return np.max(final_exploit)
  
    
def param_to_igraph(parameter, fp_indices_indicators):
    n = len(fp_indices_indicators)
    igraph = np.zeros((n, n))
    igraph[1,0] = 1
    param_pointer = 0
    for learner_idx, is_fp_idx in enumerate(fp_indices_indicators):
        if learner_idx < 2:
            continue
        
        if is_fp_idx:
            num_opponents = 1
            for opp_idx in range(learner_idx-1):
                if fp_indices_indicators[opp_idx]:
                    igraph[learner_idx, opp_idx] = 1
                    num_opponents += 1
            igraph[learner_idx, learner_idx-1] = 1
            igraph[learner_idx, :learner_idx] /= num_opponents
        else:
            next_pointer = param_pointer + learner_idx
            relevant_param = parameter[param_pointer:next_pointer]
            igraph[learner_idx, :learner_idx] = softmax(relevant_param)
            param_pointer = next_pointer
    return igraph


def calculate_param_size(fp_indices_indicators):
    param_size = 0
    for learner_idx, is_fp_idx in enumerate(fp_indices_indicators):
        if not is_fp_idx:
            param_size += learner_idx
    return param_size


def objective_with_hyperparams(parameter, game_list, fp_indices_indicators):
    igraph = param_to_igraph(parameter, fp_indices_indicators)
    return evaluate_interaction_graph_many(igraph, game_list)
    

def plot_interaction_graph(igraph, fp_indices, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(igraph, cmap="Blues")
    for learner_idx in fp_indices:
        for opp_idx in fp_indices:
            if opp_idx == learner_idx:
                break
            ax.annotate(
                'x', (opp_idx, learner_idx), c="goldenrod", fontsize=15,
                horizontalalignment='center',  verticalalignment='center'
            )
    ax.set_ylabel("Learner index")
    ax.set_xlabel("Opponent probability")
    return ax

#%% 
if __name__ == "__main__":    
    population_size = 10
    game_size = 20
    num_games = 100
    
    game_list = []
    special_graphs = { 
        alg_name: plot_metagraphs.get_metagraph(population_size, alg_name) 
        for alg_name in ["fp", "afp", "afp-fp2"]
    }
    
    for game_idx in range(num_games):
        game = np.random.normal(size=(game_size,game_size))
        game_list.append(game)
    
    
    fp_indices = list(range(0, population_size, 2))
    fp_indicators = [int(idx in fp_indices) for idx in range(population_size)]
    
    param_0 = np.random.normal(scale=1, size=calculate_param_size(fp_indicators))
    objective = partial(objective_with_hyperparams, game_list=game_list, fp_indices_indicators=fp_indicators)
    res = optimize.minimize(objective, param_0, method="BFGS", options={'gtol':1e-5, 'disp':True})
    
    optimized_igraph = param_to_igraph(res.x, fp_indicators)
    special_graphs["optimized"] = (optimized_igraph, fp_indices)
    
    #%%
    # Test on newly randomized games
    num_test_games = 2000
    test_game_list = []
    for game_idx in range(num_test_games):
        game = np.random.normal(size=(game_size,game_size))
        test_game_list.append(game)
    
    results = {
        alg_name : np.zeros(population_size - 1)
        for alg_name in special_graphs.keys()
    }
    
    std_dev = {
        alg_name : np.zeros(population_size - 1)
        for alg_name in special_graphs.keys()
    }
        
    for alg_name, (igraph, _) in special_graphs.items():
        test_results = np.zeros((num_test_games, population_size-1)) 
        for game_idx, game in enumerate(test_game_list):
            exp = evaluate_interaction_graph(igraph, game)
            test_results[game_idx] = exp
        results[alg_name] = np.mean(test_results, axis=0)
        std_dev[alg_name] = np.std(test_results, axis=0)
    
    #%%
    fig, ax = plt.subplots()
    ls = ["-","--","-.", ":"]
    
    for idx, (alg_name, mean_exp) in enumerate(results.items()):
        ax.plot(mean_exp, label=alg_name, ls=ls[idx], lw=3)
        # ax.fillbetween(mean_exp+std_dev[alg_name], mean_exp-std_dev[alg_name])
    
    ax.set_xlabel("Agent index")
    ax.set_ylabel("Exploitability")
    ax.set_title("Average performance on random matrices")
    ax.axhline(0, ls="--", c="gray", lw=1)
    ax.legend()    
    
    fig, axes = plt.subplots(ncols=len(special_graphs), figsize=(9,4))
    for idx, alg_name in enumerate(special_graphs.keys()):
        ax = axes[idx]
        plot_interaction_graph(*special_graphs[alg_name], ax=ax)
        ax.set_title(alg_name)
    
    axes[-1].set_title("Optimized metagraph")
    plt.tight_layout()