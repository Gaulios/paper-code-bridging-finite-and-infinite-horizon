import utils
import pickle
import matplotlib.pyplot as plt
import os

"""
Reproduce Figure 3 of the paper (periodic Nash equilibrium) from the saved dataset.
"""

DATA_DIR = "data"
FIG_DIR  = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

if __name__ == '__main__':

    data_path = os.path.join(DATA_DIR, "Fig3_data_paper.pkl")
    with open(data_path, 'rb') as f:
        game, K, P, L = pickle.load(f)

    utils.plot_periodic_equilibrium(game, K[1:], P, L, game.N)

    fig_path = os.path.join(FIG_DIR, "Fig3_periodic_equilibrium.pdf")
    plt.savefig(fig_path, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")
    plt.show()
    print("Finished")
