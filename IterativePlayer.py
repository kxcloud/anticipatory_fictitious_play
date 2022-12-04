import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
import ternary

import lp_solver
import games

import matplotlib.style as style
style.use('tableau-colorblind10')

def one_hot(index, length):
    array = np.zeros(length)
    array[index] = 1
    return array

def normalize_probabilities(array):
    array = np.clip(array, 0, np.inf)
    if np.sum(array) < 1e-6:
        array = array + 1e-5
    array = array / array.sum()
    return array

def get_argmaxes(array):
    max_val = np.max(array)
    return np.argwhere(np.isclose(array, max_val)).flatten(), max_val

def one_hot_argmax(array, noise=None):
    one_hot = np.zeros_like(array)
    one_hot[np.argmax(array)] = 1
    
    if noise is None:
        return one_hot
    
    noise_vector = np.random.normal(size=len(array), scale=noise)
    noisy_one_hot = normalize_probabilities(one_hot + noise_vector)
    return noisy_one_hot
    
def plot_single_player(history, empirical_history, ax=None, title=""):
    lw = 1 # line width
    
    ax = ax or plt.subplots()[1] 
    t_max, num_strategies = history.shape

    for strategy_idx in range(num_strategies):
        color = f"C{strategy_idx}"
        
        action_inds = history[:, strategy_idx].copy()
        action_inds[action_inds == 0] = np.nan
        
        ax.scatter(range(t_max), action_inds, lw=0.1, c=color, s=2)    
        ax.plot(empirical_history[:,strategy_idx], lw=lw, ls="-")
    
    ax.set_title(title)

def plot_on_triangle(iterative_player, title=None):
    # https://github.com/marcharper/python-ternary
    
    fig, axes = plt.subplots(ncols=2, figsize=(12,6))
    
    for ax in axes:
        ax.set_aspect('equal', adjustable='box')
    
    tax_p1 = ternary.TernaryAxesSubplot(ax=axes[0])
    tax_p2 = ternary.TernaryAxesSubplot(ax=axes[1])

    tax_p1.plot(np.nan_to_num(iterative_player.p1_empirical))
    tax_p2.plot(np.nan_to_num(iterative_player.p2_empirical))

    tax_p1.scatter([iterative_player.p1_probs_nash], marker="*", c="black", s=60, zorder=10)
    tax_p2.scatter([iterative_player.p2_probs_nash], marker="*", c="black", s=60, zorder=10)
    
    tax_p1.set_title("Player 1 strategies")
    tax_p2.set_title("Player 2 strategies")
    
    return axes

class IterativePlayer:
    """
    Store data from a run of a Fictitious-Play-type algorithm. 
    """
    def __init__(self, game, t_max, initial_strategy_p1, initial_strategy_p2, solve_nash=True):
        # Payoff matrix for two-player zero-sum game.
        self.game = game
        
        assert (len(initial_strategy_p1), len(initial_strategy_p2)) == self.game.shape, (
            "Strategy sizes don't match game shape"    
        )
        for strategy in [initial_strategy_p1, initial_strategy_p2]:
            assert np.isclose(np.sum(strategy), 1, atol=1e-4), f"Probs must sum to 1. {strategy}"
        
        self.p1_response = np.zeros((t_max, game.shape[0]))
        self.p2_response = np.zeros((t_max, game.shape[1]))
        self.p1_response[0,:] = initial_strategy_p1
        self.p2_response[0,:] = initial_strategy_p2

        self.p1_empirical = self.p1_response.copy()
        self.p2_empirical = self.p2_response.copy()
        
        self.worst_case_payoff = np.zeros((t_max, 2))
        self.worst_case_payoff[0,:] = self.get_worst_case_payoffs(initial_strategy_p1, initial_strategy_p2)
        
        self.total_compute = np.zeros((t_max,2))
        self.total_compute[0,:] = 1
        
        self.is_ephemeral = np.zeros(t_max, dtype=bool)
        
        # Minimax Theorem (von Neumann, 1928): the set of Nash equilibria
        # is { (x*, y*) : x* is maximin for P1, y* is maximin for P2 }.
        # Informally, 2P0S => Nash = Maximin = Minimax.
        if solve_nash:
            self.p2_probs_nash, info = lp_solver.solve_game_half(game)
            self.p1_probs_nash, _   = lp_solver.solve_game_half(-game.T)
            self.value = info["res"]["fun"]
        
        self.t = 1
    
    def add_strategies(
            self, p1_strategy, p2_strategy, p1_compute=1, p2_compute=1, strategy_is_ephemeral=False
        ):
        """ 
        The strategies here are the ones added by a FP-like algorithm.
        
        This function simply updates the data
        """
        assert (len(p1_strategy), len(p2_strategy)) == self.game.shape, (
            "Strategy sizes don't match game shape"    
        )
        assert np.isclose(np.sum(p1_strategy), 1, atol=1e-4), f"Probs must sum to 1. {p1_strategy}"
        assert np.isclose(np.sum(p2_strategy), 1, atol=1e-4), f"Probs must sum to 1. {p2_strategy}"
        
        self.is_ephemeral[self.t] = strategy_is_ephemeral
        strats_to_keep = ~self.is_ephemeral[:self.t+1]
        strats_to_keep[-1] = True
        
        self.p1_response[self.t,:] = p1_strategy
        self.p2_response[self.t,:] = p2_strategy
                
        self.p1_empirical[self.t,:] = np.mean(self.p1_response[:self.t+1,][strats_to_keep], axis=0)
        self.p2_empirical[self.t,:] = np.mean(self.p2_response[:self.t+1,][strats_to_keep], axis=0)
        
        self.worst_case_payoff[self.t] = self.get_worst_case_payoffs(
            self.p1_empirical[self.t], self.p2_empirical[self.t]
        )
        
        self.total_compute[self.t,:] = self.total_compute[self.t-1,:] + [p1_compute, p2_compute]
        
        self.t += 1
    
    def get_worst_case_payoffs(self, p1_strategy, p2_strategy):
        worst_case_p1 = np.min(p1_strategy @ self.game)
        worst_case_p2 = -np.max(self.game @ p2_strategy)
        return worst_case_p1, worst_case_p2
    
    def plot(self, title="", players_to_plot=[0,1], figsize=(4,3)):   
        
        fig, axes = plt.subplots(
            nrows=2, ncols=len(players_to_plot), sharex=True, 
            sharey="row", figsize=figsize, squeeze=False
        )
        
        if 0 in players_to_plot:
            plot_single_player(self.p1_response, self.p1_empirical, axes[0,0], "Player 1")
            axes[1,0].axhline(self.value, label="value of game", ls="--", c="black", lw=0.8)
        if 1 in players_to_plot:
            plot_single_player(self.p2_response, self.p2_empirical, axes[0,1], "Player 2")
            axes[1,1].axhline(-self.value, label="value of game", ls="--", c="black", lw=0.8)
        
        for player_idx in players_to_plot:
            ax = axes[1,player_idx]
            ax.plot(self.worst_case_payoff[:,player_idx], lw=2, c="grey", label="average strategy")
        
        for player_idx, nash_probs in enumerate([self.p1_probs_nash, self.p2_probs_nash]):
            if player_idx in players_to_plot:
                for action_idx, prob in enumerate(nash_probs):
                    color = f"C{action_idx}"
                    axes[0, player_idx].axhline(prob, ls="--", c=color, lw=1)
                    
        axes[0,0].set_ylabel("Strategy probabilities")
        axes[1,0].set_ylabel("Worst-case payoff")
        axes[-1,0].set_xlabel("Timestep")
        
        axes[1,max(players_to_plot)].legend()
        plt.suptitle(title)       
        plt.tight_layout()
        return axes

def run_fp(
        game, t_max, initial_strategy_p1, initial_strategy_p2, tiebreak_fn,
        symmetric=False, solve_nash=False
    ):
    if symmetric:
       assert np.all(game==-game.T), "Payoff matrix must be antisymmetric"
        
    play = IterativePlayer(game, t_max, initial_strategy_p1, initial_strategy_p2, solve_nash=solve_nash)
        
    for t in range(1, t_max):
        p1_argmaxes, _ = get_argmaxes(play.game @ play.p2_empirical[t-1])
        p1_argmax = tiebreak_fn(p1_argmaxes)
        
        if symmetric:
            p2_argmax = p1_argmax
        else:
            p2_argmaxes, _ = get_argmaxes(-play.game.T @ play.p1_empirical[t-1])
            p2_argmax = tiebreak_fn(p2_argmaxes)
        
        play.add_strategies(
            one_hot(p1_argmax, game.shape[0]), one_hot(p2_argmax, game.shape[1])
        )
    
    return play

def run_afp(
        game, t_max, initial_strategy_p1, initial_strategy_p2, tiebreak_fn,
        symmetric=False, solve_nash=False, num_initial_fp_steps=0
    ):
    if symmetric:
        assert np.all(game==-game.T), "Payoff matrix must be antisymmetric"
        
    play = IterativePlayer(game, t_max, initial_strategy_p1, initial_strategy_p2, solve_nash=solve_nash)
            
    for t in range(1, t_max):
        if t <= num_initial_fp_steps: 
            p1_argmaxes, _ = get_argmaxes(play.game @ play.p2_empirical[t-1])
            p1_argmax = tiebreak_fn(p1_argmaxes)
            
            if symmetric:
                p2_argmax = p1_argmax
            else:
                p2_argmaxes, _ = get_argmaxes(-play.game.T @ play.p1_empirical[t-1])
                p2_argmax = tiebreak_fn(p2_argmaxes)
            
            play.add_strategies(
                one_hot(p1_argmax, game.shape[0]), one_hot(p2_argmax, game.shape[1])
            )
            
        else:
            p1_payoffs = play.game @ play.p2_empirical[t-1]*t
            p2_payoffs = -play.game.T @ play.p1_empirical[t-1]*t
            
            p2_argmaxes_a, _ = get_argmaxes(p2_payoffs)
            p2_br = one_hot(tiebreak_fn(p2_argmaxes_a), game.shape[1])
            
            p1_argmaxes_b, _ = get_argmaxes(p1_payoffs + play.game @ p2_br)
            
            p1_ar = one_hot(tiebreak_fn(p1_argmaxes_b), game.shape[0])
            
            if symmetric:
                p2_ar = p1_ar
            else:
                p1_argmaxes_a, _ = get_argmaxes(p1_payoffs)
                p1_br = one_hot(tiebreak_fn(p1_argmaxes_a), game.shape[0])
                p2_argmaxes_b, _ = get_argmaxes(p2_payoffs + -play.game.T @ p1_br)
                p2_ar = one_hot(tiebreak_fn(p2_argmaxes_b), game.shape[1])
            
            play.add_strategies(
                p1_ar, p2_ar, p1_compute=2, p2_compute=2
            )
        
    return play

def run_afp_parity(
        game, t_max, initial_strategy_p1, initial_strategy_p2, tiebreak_fn,
        symmetric=False, solve_nash=False, num_initial_fp_steps=0
    ):
    """ Uses one br per timestep.  """
    
    if symmetric:
        assert np.all(game==-game.T), "Payoff matrix must be antisymmetric"
        
    play = IterativePlayer(game, t_max, initial_strategy_p1, initial_strategy_p2, solve_nash=solve_nash)
    
    offset = num_initial_fp_steps % 2
    
    for t in range(1, t_max):
        eligible_indices = [i for i in range(t) if i % 2 == offset or i <= num_initial_fp_steps or i==t-1 ]
        p2_empirical = play.p2_response[eligible_indices,:].mean(axis=0)
        
        # if not np.isclose(p2_empirical, play.p2_empirical[play.t-1]).all():
        #     breakpoint()
        
        p1_argmaxes, _ = get_argmaxes(play.game @ p2_empirical)
        p1_argmax = tiebreak_fn(p1_argmaxes)
        
        if symmetric:
            p2_argmax = p1_argmax
        else:
            p1_empirical = play.p1_response[eligible_indices,:].mean(axis=0)
            p2_argmaxes, _ = get_argmaxes(-play.game.T @ p1_empirical)
            p2_argmax = tiebreak_fn(p2_argmaxes)
        
        play.add_strategies(
            one_hot(p1_argmax, game.shape[0]), one_hot(p2_argmax, game.shape[1]),
            strategy_is_ephemeral = (t % 2 != offset) and (t > num_initial_fp_steps)
        )
    
    return play   


def anticipatory_response(play, k):
    p1_payoffs = play.game @ play.p2_empirical[play.t-1]*play.t
    p2_payoffs = -play.game.T @ play.p1_empirical[play.t-1]*play.t
    
    for _ in range(k+1):
        p1_br = one_hot_argmax(p1_payoffs)
        p2_br = one_hot_argmax(p2_payoffs)
        
        p1_payoffs = p1_payoffs + play.game @ p2_br
        p2_payoffs = p2_payoffs + -play.game.T @ p1_br
    
    return p1_br, p2_br    

def run_afp_general(
        game, t_max, initial_strategy_p1, initial_strategy_p2, steps_to_anticipate=0, noise=None
):
    play = IterativePlayer(game, t_max, initial_strategy_p1, initial_strategy_p2)
            
    for t in range(1, t_max):
        p1_ar, p2_ar = anticipatory_response(play, steps_to_anticipate)
        play.add_strategies(p1_ar, p2_ar, p1_compute=steps_to_anticipate+1, p2_compute=steps_to_anticipate+1)
        
    return play
    
if __name__ == "__main__":
    # seed = np.random.choice(10000)
    # print(f"Seed: {seed}")
    # np.random.seed(seed)
    
    # game_name = "RPS"
    # game_name = "Matching Pennies"
    # game_name = "Biased RPS"
    # game_name = "weakRPS"
    # game_name = "RPS + safe R"
    game_name = "RPS + safe R asym"
    # game_name = "Random game"
    # game_name = "RPS with mixed moves"
    # game_name = "RPS Abstain"
    # game_name = "Cyclic game"
    # game_name = "Cyclic game (n=50)"
    # game_name = "Albert's RPS + safe R"
    # game_name = "Identity (50)"
    # game_name = "Transitive game"
    
    game = games.game_dict[game_name]
    
    initial_strategy_p1 = one_hot(0, game.shape[0]) #np.ones(game.shape[0]) / game.shape[0]
    initial_strategy_p2 = one_hot(0, game.shape[1]) #np.ones(game.shape[1]) / game.shape[1]
    t_max = 500
        
    # FICTITIOUS PLAY
    play = IterativePlayer(game, t_max, initial_strategy_p1, initial_strategy_p2, solve_nash=True)
        
    for t in range(1, t_max):
        p1_argmaxes, _ = get_argmaxes(play.game @ play.p2_empirical[t-1])
        p2_argmaxes, _ = get_argmaxes(-play.game.T @ play.p1_empirical[t-1])
        p1_argmax = np.random.choice(p1_argmaxes)
        p2_argmax = np.random.choice(p2_argmaxes)
        play.add_strategies(
            one_hot(p1_argmax, game.shape[0]), one_hot(p2_argmax, game.shape[1])
        )
            
    print("done")
    axes = play.plot(title=f"Fictitious Play: {game_name}", players_to_plot=[0], figsize=(8,6))
    # axes[-1,-1].plot([max(-2/t**0.5,np.min(game)) for t in range(1,t_max+1)])
    # axes[-1,-1].plot([max(-1/t**0.5,np.min(game)) for t in range(1,t_max+1)])
    # axes[-1,-1].plot([max(-2/t,np.min(game)) for t in range(1,t_max+1)])
    actions_played = np.nonzero(play.p1_response)[1]
    print(actions_played)
    
    counts = []
    last_a = None
    count = 0
    for a in actions_played:
        if a != last_a:
            counts.append(count)
            count = 1
        else:
            count += 1
        last_a = a
    print("streaks           ", counts)
    print("total_to_increment", np.cumsum(counts))
    
    
    # print("Actions played:")
    # play_counts = (play.p1_empirical*np.array(range(1, t_max+1))[:,None]).round()
    # print("Play counts:")
    # print(play_counts)
    # print("Exploitability x t:")
    # print(np.max(game @ play_counts.T, axis=0))
    
    # # plot_on_triangle(play)
    
    # # ANTICIPATORY FICTITIOUS PLAY
    # play = IterativePlayer(game, t_max, initial_strategy_p1, initial_strategy_p2)
    
    # for t in range(1, t_max):
    #     noise = None
    #     p1_payoffs = play.game @ play.p2_empirical[t-1]*t
    #     p2_payoffs = -play.game.T @ play.p1_empirical[t-1]*t
        
    #     p1_br = one_hot_argmax(p1_payoffs, noise=noise)
    #     p2_br = one_hot_argmax(p2_payoffs, noise=noise)
        
    #     p1_ar = one_hot_argmax(p1_payoffs + play.game @ p2_br, noise=noise)
    #     p2_ar = one_hot_argmax(p2_payoffs + -play.game.T @ p1_br, noise=noise)
        
    #     # p1_ar2 = one_hot_argmax(p1_payoffs + play.game @ p2_br + play.game @ p2_ar, noise=noise)
    #     # p2_ar2 = one_hot_argmax(p2_payoffs + -play.game.T @ p1_br -play.game.T @ p1_ar, noise=noise)
        
    #     play.add_strategies(
    #         p1_ar, p2_ar
    #     )
        
    # axes = play.plot(title=f"Anticipatory Fictitious Play: {game_name}", players_to_plot=[0], figsize=(8,6))
    # axes[-1,-1].plot([max(-4/t,np.min(game)) for t in range(1,t_max+1)])
    # axes[-1,-1].set_ylim([-0.2,0])
    # print("Actions played:")
    # play_counts = (play.p1_empirical*np.array(range(1, t_max+1))[:,None]).round()
    # print("Play counts:")
    # print(play_counts)
    # print("Exploitability x t:")
    # print(np.max(game @ play_counts.T, axis=0))
    # # plot_on_triangle(play)