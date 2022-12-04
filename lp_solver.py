import numpy as np
from scipy.optimize import linprog

def solve_game_half(A):
    """
    Given player 1's payoff matrix for a two-player zero-sum game,
    computes Nash Equilibrium probabilities for player 2 by solving
    a linear program.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
    """
    m, n = A.shape

    # The linear program has 1+n variables:
    # one for the value of the game and one per player 2 action.

    # Objective
    c = np.zeros(n + 1)
    c[0] = 1

    # Inequality constraints
    A_ub = np.zeros((m, n + 1))
    A_ub[:, 0] = -1
    for j in range(m):
        A_ub[j, 1:] = A[j, :]

    b_ub = np.zeros(m)

    # Equality constraints
    A_eq = np.ones(shape=(1, n + 1))
    A_eq[0, 0] = 0
    b_eq = np.array([1])

    # Bounds
    bounds = np.zeros((n + 1, 2))  # Probabilities must be greater than 0
    bounds[:, 1] = 1  # Probabilities must be less than 1
    bounds[0, :] = None  # the value of the game is unbounded

    res = linprog(
        c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, 
        method="interior-point", options={"lstsq":True, "tol": 1e-8},
    )
    p2_probs = res.x[1:]

    info = {"res": res}
    return p2_probs, info
    
if __name__ == "__main__":
    rps_safe = np.array([[0,-1,1,0],[1,0,-1,0.49],[-1,1,0,-1/2],[0,-0.49,1/2,0]])
    rps_safe = np.array([[0,-1,1,0]])

    p2_probs, info = solve_game_half(rps_safe)
    p1_probs, info = solve_game_half(-rps_safe.T)
    
    print(rps_safe)
    print(f"P1 probs: {p1_probs.round(2)}")
    print(f"P2 probs: {p2_probs.round(2)}")