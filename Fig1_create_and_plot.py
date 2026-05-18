import utils
import numpy as np
import pickle
import os

"""
Convergence statistics of the backward Riccati recursion across random LQ games.

For each (state dimension n, number of players N) we sample 10^4 random games
and classify the asymptotic behavior of the recursion as:
  - convergence to a static Nash equilibrium,
  - convergence to a periodic Nash equilibrium,
  - failure to converge.
Results are displayed as three side-by-side heat maps.
"""

DATA_DIR = "data"
FIG_DIR  = "figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

if __name__ == '__main__':

    # Grid parameters
    min_n, max_n = 1, 8
    min_N, max_N = 2, 8
    T_horizon    = 400
    num_games    = 10 ** 4

    static   = np.zeros((max_n + 1 - min_n, max_N + 1 - min_N))
    periodic = np.zeros_like(static)
    not_conv = np.zeros_like(static)

    for n in range(min_n, max_n + 1):
        print(f"--- n = {n} ---")
        d = n   # action dimension equals state dimension

        for N in range(min_N, max_N + 1):
            print(f"  N = {N}")

            for _ in range(num_games):
                game = utils.generate_random_game(n, d, N)
                K, P = utils.run_backward_riccati(T_horizon, game, game.P)
                conv = utils.detect_convergence_type(K, T_horizon, game.N)

                row, col = n - min_n, N - min_N
                if conv == 1:
                    static[row, col]   += 1
                elif conv > 1:
                    periodic[row, col] += 1
                else:
                    not_conv[row, col] += 1

            row, col = n - min_n, N - min_N
            print(f"    Static:     {static[row, col] / num_games:.3f}")
            print(f"    Periodic:   {periodic[row, col] / num_games:.3f}")
            print(f"    No conv.:   {not_conv[row, col] / num_games:.3f}")

    static   /= num_games
    periodic /= num_games
    not_conv /= num_games

    # Save data
    data = (static, periodic, not_conv, min_n, min_N, max_n, max_N)
    data_path = os.path.join(DATA_DIR, "Fig1_data.pkl")
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {data_path}")

    # Plot and save figure
    import matplotlib.pyplot as plt
    utils.plot_convergence_heatmaps(static, periodic, not_conv, min_n, min_N, max_n, max_N)
    fig_path = os.path.join(FIG_DIR, "Fig1_convergence_heatmaps.pdf")
    plt.savefig(fig_path, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")
    plt.show()
    print("Finished")
