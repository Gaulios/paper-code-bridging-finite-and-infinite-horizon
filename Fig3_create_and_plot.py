import utils
import numpy as np
import pickle
import os

"""
Search for a random LQ game whose backward Riccati recursion converges to a
periodic orbit of length L >= 2, where at least one policy in the cycle has
spectral radius > 1 (i.e., is individually destabilizing).

The resulting periodic Nash equilibrium is then saved and visualized:
  - Top panel: ||P^i_t - P^i_0||_F over time.
  - Bottom panel: spectral radius of A^cl_t over time.
"""

DATA_DIR = "data"
FIG_DIR  = "figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

if __name__ == '__main__':

    # Game parameters
    n, d, N = 3, 2, 4
    T = 300
    min_period = 2

    game, K, P, L = None, None, None, 1
    found = False
    for it in range(10 ** 4):
        print(f"Iteration {it + 1}")
        game = utils.generate_random_game(n, d, N)
        K, P = utils.run_backward_riccati(T, game, game.P)
        L = utils.detect_convergence_type(K, T, N)

        if L >= min_period:
            # Check whether any policy in the cycle is individually destabilizing
            for l in range(L):
                C = game.A.copy()
                for i in range(N):
                    C -= game.B[i] @ K[l][i]
                if np.max(np.abs(np.linalg.eigvals(C))) >= 2:
                    found = True
                    break

        if found:
            break

    print(f"Found a game: period L={L}, individually destabilizing policy: {found}")

    if L > 1 and found:
        # Save data
        data = (game, K, P, L)
        data_path = os.path.join(DATA_DIR, "Fig3_data.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {data_path}")

        # Plot and save figure
        import matplotlib.pyplot as plt
        utils.plot_periodic_equilibrium(game, K, P, L, N)
        fig_path = os.path.join(FIG_DIR, "Fig3_periodic_equilibrium.pdf")
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
        plt.show()

    print("Finished")
