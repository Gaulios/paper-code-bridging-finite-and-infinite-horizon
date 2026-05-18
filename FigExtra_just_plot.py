import utils
import pickle
import matplotlib.pyplot as plt
import os

"""
Reproduce the policy-evolution figure from the saved dataset.
"""

DATA_DIR = "data"
FIG_DIR  = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

if __name__ == '__main__':

    data_path = os.path.join(DATA_DIR, "FigExtra_data_paper.pkl")
    with open(data_path, 'rb') as f:
        game, K, P, L, K_grad = pickle.load(f)

    time_steps = 30
    utils.plot_policy_evolution(K[:time_steps], K_grad, N=game.N, zoom=False)

    fig_path = os.path.join(FIG_DIR, "FigExtra_policy_evolution.pdf")
    plt.savefig(fig_path, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")
    plt.show()
    print("Finished")
