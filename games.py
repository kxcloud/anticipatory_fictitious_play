import numpy as np

def symmetrize_game(game):
    return (game - game.T)/2
    
def get_rps_with_mixed_moves(bonus=0):
    rps = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
    
    moves = [[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]]
    bonuses = [0,0,0, bonus, bonus, bonus]
    
    rps_with_mixed_moves = np.zeros((6,6))
    for i, (move_1, bonus_1) in enumerate(zip(moves,bonuses)):
        for j, (move_2, bonus_2) in enumerate(zip(moves,bonuses)):
            rps_with_mixed_moves[i,j] = move_1 @ rps @ move_2 + bonus_1 - bonus_2
    return rps_with_mixed_moves
    
def get_rps_abstain(bonus=0):
    rps = np.array([[0,-1,1],[1,0,-1],[-1,1,0]])
    moves = [[1,0,0],[0,1,0],[0,0,1],[0,0,0]]
    bonuses = [0,0,0, bonus]
    
    rps_abstain = np.zeros((4,4))
    for i, (move_1, bonus_1) in enumerate(zip(moves,bonuses)):
        for j, (move_2, bonus_2) in enumerate(zip(moves,bonuses)):
            rps_abstain[i,j] = move_1 @ rps @ move_2 + bonus_1 - bonus_2
    return rps_abstain

def get_matching_pennies_abstain(bonus):
    return np.array([
        [1,-1, -bonus],
        [-1,1, -bonus],
        [bonus, bonus, 0]
    ])

def get_cyclic_game(n):
    """
    Return a payoff matrix for a cyclic game. 
        3 -> RPS
    """
    game = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(n):
            if i == (j-1) % n:
                game[i,j] += -1
            if i == (j+1) % n:
                game[i,j] += 1
            
    return game
 
def get_transitive_game_old(n):
    game = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(i):
            if i-1 == j:
                game[i,j] = 2
                game[j,i] = -2
            elif i-2 == j:
                game[i,j] = 1
                game[j,i] = -1
    return game

def get_transitive_game(n):
    game = np.zeros(shape=(n,n))
    for i in range(n):
        for j in range(i):
            if i-1 == j:
                game[i,j] = (n-i+2)/n
                game[j,i] = -(n-i+2)/n
    return game

rand_seed = np.random.seed(np.random.choice(1000))

# np.random.seed(2) # Fix seed only for game generation

game_dict = {
    "Matching Pennies" : np.array([[1,-1],[-1,1]]),
    "Matching Pennies Abstain" : get_matching_pennies_abstain(bonus=0.05),
    "RPS" : np.array([[0,-1,1],[1,0,-1],[-1,1,0]]),
    "Biased RPS" : np.array([[0,-2,1],[2,0,-1],[-1,1,0]]),
    "Biased RPS asym" : np.array([[0,-1,2],[1,0,-1],[-1,1,0]]),
    "weakRPS" : np.array([[0,-1,1e-1],[1,0,-1],[-1e-1,1,0]]),
    "RPS + safe R" : np.array([[0,-1,1,0],[1,0,-1,0.1],[-1,1,0,-0.9],[0,-0.1,0.9,0]]),
    "RPS + safe R asym" : np.array([[0,-1,1],[1,0,-1],[-1,1,0],[0,-0.1,0.9]]),
    "RPS Abstain": get_rps_abstain(bonus=0.05),
    "Random game 1" : np.array([[ 1.62, -0.61, -0.53],
                                [-1.07,  0.87, -2.3 ],
                                [ 1.74, -0.76,  0.32]]),
    "Random game 2" : np.array([[-0.42, -0.06, -2.14],
                                [ 1.64, -1.79, -0.84],
                                [ 0.5,  -1.25, -1.06]]),
    "Random symmetric game" : symmetrize_game(np.random.normal(size=(3,3))),
    "RPS with mixed moves" : get_rps_with_mixed_moves(bonus=0.1),
    "Albert's RPS + safe R": np.array(
        [
            [ 0, -1,  1, 0.0],
            [ 1,  0, -1, 0.88],
            [-1,  1,  0, -0.9],
            [ 0.0, -0.88, 0.9, 0.0],
        ]),
    "Cyclic game" : get_cyclic_game(9),
    "Cyclic game (n=20)" : get_cyclic_game(20),
    "Transitive game (n=20)" : get_transitive_game(20),
    "Cyclic game (n=7)" : get_cyclic_game(7),
    "Transitive game (n=7)" : get_transitive_game(7),
    "Identity (3)" : np.eye(3),
    "Identity (10)" : np.eye(10),
    "Identity (50)" : np.eye(50),
    "Starcraft Macro" : 
        np.array([[0.5,0.65,0.15],[0.35,0.5,0.55],[0.85,0.45,0.5]]),
    "RPSLS":
        np.array([[ 0., -1.,  1.,  1., -1.],
         [ 1.,  0., -1., -1.,  1.],
         [-1.,  1.,  0.,  1., -1.],
         [-1.,  1., -1.,  0.,  1.],
         [ 1., -1.,  1., -1.,  0.]])
  }

np.random.seed(rand_seed)

if __name__ == "__main__":
    n=7
    c_5 = get_cyclic_game(n)
    t_5 = get_transitive_game(n)
    
    import matplotlib.pyplot as plt
    plt.rcParams["figure.dpi"] = 150
    
    fig, axes = plt.subplots(ncols=2, figsize=(7, 4))
    titles = ["$C^n$, " f"$n=${n}", "$T^n$, " f"$n=${n}"]
    
    for (game, ax, title) in zip([c_5,t_5], axes, titles):
        im = ax.imshow(game, vmin=-1, vmax=1, cmap="RdYlBu")
        ax.set_title(title)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.5])
    fig.colorbar(im, cax=cbar_ax)
    
    
    plt.show()