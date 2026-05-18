import numpy as np
import utils
import pickle

"""
Scalar two-player game: convergence map of the finite-horizon Riccati recursion.

For a grid of terminal costs (P1, P2) we run the backward Riccati recursion and
record which infinite-horizon Nash equilibrium is reached.  We also plot a few
sample trajectories.

The infinite-horizon NEs are computed using the closed-form characterization from:
  Salizzoni, G., Ouhamma, R., & Kamgarpour, M. (2024).
  "Nash equilibria in scalar discrete-time linear quadratic games."
  2025 European Control Conference (ECC).
"""

import os
DATA_DIR = "data"
FIG_DIR  = "figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)


### Scalar-case helpers ######################################################

def compute_cost(a, qi, ri, ki, kj):
    """Infinite-horizon cost for agent i under Nash policies (ki, kj)."""
    denominator = 1 - (a - ki - kj) ** 2
    if abs(denominator) < 1e-8:
        return np.inf
    return (qi + ri * ki ** 2) / denominator

def compute_infinite_horizon_nash(a, q1, q2, r1, r2):
    """Return arrays nash_eqs (M,2) and costs (M,2) for all real NEs."""
    k2_roots = polynomial_roots(a, q1, q2, r1, r2)
    nash_eqs, costs = [], []
    for root in k2_roots:
        if np.isreal(root) and 0 < root.real < a:
            k2 = root.real
            k1 = best_response(a, q1, r1, k2)
            costs.append([compute_cost(a, q1, r1, k1, k2),
                          compute_cost(a, q2, r2, k2, k1)])
            nash_eqs.append([k1, k2])
    return np.array(nash_eqs), np.array(costs)

def compute_convergence_map(samples, min_P, max_P, T, game, nash_eqs):
    """
    For a (samples x samples) grid of terminal costs (P1, P2), run the backward
    Riccati recursion and record which NE is reached.
    """
    N = game.N
    convergence_map = np.zeros((samples, samples))
    terminal_cost = np.zeros((N, 1, 1))
    for i in range(samples):
        if i % 10 == 0:
            print(f"  Progress: {i}/{samples}")
        terminal_cost[0] = (max_P * i) / samples + min_P
        for j in range(samples):
            terminal_cost[1] = (max_P * j) / samples + min_P
            K, Pt = utils.run_backward_riccati(T, game, terminal_cost)
            convergence_map[j, i] = index_ne_reached(K, T, nash_eqs)
    return convergence_map

def compute_trajectories(initial_pairs, T, game):
    """
    Run the backward Riccati recursion for each initial terminal cost pair.
    Returns a list of arrays of shape (T+1, 2): [P1_t, P2_t].
    """
    trajectories = []
    for P_init in initial_pairs:
        K, Pt = utils.run_backward_riccati(T, game, P_init)
        traj = np.zeros((T + 1, 2))
        traj[:, 0] = Pt[:, 0, 0, 0]
        traj[:, 1] = Pt[:, 1, 0, 0]
        trajectories.append(traj)
    return trajectories

def best_response(a, qi, ri, kj):
    """Best response of agent i to opponent policy kj."""
    temp = ri * (1 - (a - kj) ** 2) + qi
    sqrt_term = np.sqrt(temp ** 2 + 4 * ri * ((a - kj) ** 2) * qi)
    if sqrt_term + temp == 0:
        return 0.0
    return (2 * (a - kj) * qi) / (sqrt_term + temp)

def polynomial_roots(a, q1, q2, r1, r2):
    """
    Roots of the degree-5 characteristic polynomial whose real roots in (0, a)
    correspond to the second player's NE policies.
    """
    v1 = a * (r1 ** 2) * (r2 ** 2)
    v2 = (-2.5 * a**2 * r1**2 * r2**2 + q2 * r1**2 * r2
          - q1 * r1 * r2**2 + r1**2 * r2**2)
    v3 = (2 * a**3 * r1**2 * r2**2 - 2*a*q2*r1**2*r2
          + 2*a*q1*r1*r2**2 - 2*a*r1**2*r2**2)
    v4 = (-0.5*a**4*r1**2*r2**2 + a**2*q2*r1**2*r2 - a**2*q1*r1*r2**2
          + a**2*r1**2*r2**2 + 0.5*q2**2*r1**2 - 0.5*q1**2*r2**2
          - q1*r1*r2**2 - 0.5*r1**2*r2**2)
    v5 = -a * q2**2 * r1**2
    v6 = 0.5 * a**2 * q2**2 * r1**2
    return np.roots([v1, v2, v3, v4, v5, v6])

def index_ne_reached(K, T, nash_eqs):
    """
    Return a signed index indicating which NE the recursion converged to.
    The sign encodes the NE index; the magnitude encodes the convergence time.
    """
    time = T - 1
    while time > -1:
        k0 = K[time, 0, 0, 0]
        k1 = K[time, 1, 0, 0]
        minimum = abs(k0 - nash_eqs[0, 0]) + abs(k1 - nash_eqs[0, 1])
        index = 1
        for i in range(1, nash_eqs.shape[0]):
            dist = abs(k0 - nash_eqs[i, 0]) + abs(k1 - nash_eqs[i, 1])
            if dist < minimum:
                index = i + 1
                minimum = dist
        if minimum < 0.001:
            return (index - 2) * (T - time)
        time -= 1
    print("Recursion did not converge to any NE.")
    return 0


### Main #####################################################################

if __name__ == '__main__':

    # Game parameters
    N = 2
    a = 5
    A = np.array([[a]])
    b1, b2 = 1, 1
    B = np.array([[[b1]], [[b2]]])
    q1, r1 = 1, 1
    q2, r2 = 1, 2
    Q = np.array([[[q1]], [[q2]]])
    R = np.array([[[r1]], [[r2]]])
    P = np.zeros((N, 1, 1))
    game = utils.GameData(1, 1, N, A, B, Q, R, P)
    T = 100

    # Compute infinite-horizon NEs
    nash_eqs, costs = compute_infinite_horizon_nash(a, q1, q2, r1, r2)
    for idx, (eq, cost) in enumerate(zip(nash_eqs, costs)):
        print(f"NE #{idx + 1}: policies = {eq},  costs = {cost}")
    print("-" * 50)

    # Classify each NE as attractor / saddle via the Jacobian
    print("Fixed-point classification of the Riccati map at each NE:")
    for idx, cost in enumerate(costs):
        J = utils.jacobian_at_fixed_point(cost[0], cost[1], game)
        eigvals = np.linalg.eigvals(J)
        label = utils.classify_fixed_point(eigvals)
        print(f"  NE #{idx + 1}: |eigenvalues| = {np.round(np.abs(eigvals), 6)}  ->  {label}")
    print("-" * 50)

    # Compute the convergence map
    samples = 1000
    min_P, max_P = 0, 50
    print("Computing convergence map ...")
    convergence_map = compute_convergence_map(samples, min_P, max_P, T, game, nash_eqs)

    # Compute sample trajectories
    initial_pairs = np.array([
        [[[5]],  [[2]]],
        [[[20]], [[45]]],
        [[[8]],  [[48]]],
        [[[31]], [[35]]],
    ], dtype=float)
    trajectories = compute_trajectories(initial_pairs, T, game)

    # Save data
    output_data = {
        "convergence_map": convergence_map,
        "trajectories":    trajectories,
        "costs":           costs,
        "nash_eqs":        nash_eqs,
        "T":               T,
        "minP":            min_P,
        "maxP":            max_P,
    }
    data_path = os.path.join(DATA_DIR, "Fig2_data.pkl")
    with open(data_path, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"Data saved to {data_path}")

    # Plot and save figure
    import matplotlib.pyplot as plt
    utils.plot_convergence_map(convergence_map, trajectories, costs, nash_eqs, T, min_P, max_P)
    fig_path = os.path.join(FIG_DIR, "Fig2_convergence_map.pdf")
    plt.savefig(fig_path, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")
    plt.show()
    print("Finished")
