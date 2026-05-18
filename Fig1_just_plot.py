import utils
import pickle
import matplotlib.pyplot as plt
import os

"""
Reproduce Figure 1 of the paper (convergence heatmaps) from the saved dataset.
"""

DATA_DIR = "data"
FIG_DIR  = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

if __name__ == '__main__':

    data_path = os.path.join(DATA_DIR, "Fig1_data_paper.pkl")
    with open(data_path, 'rb') as f:
        static, periodic, not_conv, min_n, min_N, max_n, max_N = pickle.load(f)

    utils.plot_convergence_heatmaps(static, periodic, not_conv, min_n, min_N, max_n, max_N)

    fig_path = os.path.join(FIG_DIR, "Fig1_convergence_heatmaps.pdf")
    plt.savefig(fig_path, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")
    plt.show()
    print("Finished")
