import utils
import numpy as np
import pickle
import os

"""
Search for a random LQ game whose backward Riccati recursion does not converge,
then use policy gradient to find a stationary Nash equilibrium nearby.
The evolution of the policies is plotted in the (K[1,1], K[1,2]) plane.

Note: policy gradient has no convergence guarantees here; parameters may need tuning.
"""

DATA_DIR = "data"
FIG_DIR  = "figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

if __name__ == '__main__':

    # Game parameters
    n, d, N = 2, 1, 2
    T = 300

    # Search for a game with non-convergent recursion
    game, K, P, L = None, None, None, None
    for it in range(1000):
        print(f"Iteration {it + 1}")
        game = utils.generate_random_game(n, d, N)
        K, P = utils.run_backward_riccati(T, game, game.P)
        L = utils.detect_convergence_type(K, T, N)
        if L == -1:
            break

    print(f"Found a game with non-convergent recursion (L={L}).")
    print("-" * 50)

    # Policy gradient to find a stationary NE
    eta = 1e-2
    num_iterations = int(1e3)
    K_best = None
    norm = 1000.0

    # Warm-start from the average of the first few iterates
    K_init = np.mean(K[:20], axis=0)

    for attempt in range(4):
        print(f"Gradient descent: eta={eta}, iterations={num_iterations}")
        K_init_run = K_init if K_best is None else K_best
        K_grad, P_grad, K_best, grad = utils.compute_gradient_descent(
            game, K_init_run, eta, num_iterations
        )
        norm = sum(np.linalg.norm(grad[-1, i], 'fro') for i in range(N))
        if norm <= N * 1e-2:
            break
        eta /= 2
        num_iterations *= 10

    print(f"Final gradient norm: {norm:.6f}")
    print("-" * 50)

    # Save data
    data = (game, K, P, L, K_grad[-1])
    data_path = os.path.join(DATA_DIR, "FigExtra_data.pkl")
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {data_path}")

    # Plot and save figure
    import matplotlib.pyplot as plt
    time_steps = 30
    utils.plot_policy_evolution(K[:time_steps], K_grad[-1], N, zoom=False)
    fig_path = os.path.join(FIG_DIR, "FigExtra_policy_evolution.pdf")
    plt.savefig(fig_path, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")
    plt.show()
    print("Finished")
