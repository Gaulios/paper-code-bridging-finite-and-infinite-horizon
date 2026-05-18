import utils
import pickle
import matplotlib.pyplot as plt
import os

"""
Reproduce Figure 2 of the paper (scalar convergence map) from the saved dataset.
"""

DATA_DIR = "data"
FIG_DIR  = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

if __name__ == '__main__':

    data_path = os.path.join(DATA_DIR, "Fig2_data_paper.pkl")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    convergence_map = data["convergence_map"]
    trajectories    = data["trajectories"]
    costs           = data["costs"]
    nash_eqs        = data["nash_eqs"]
    T               = data["T"]
    min_P           = data["minP"]
    max_P           = data["maxP"]

    utils.plot_convergence_map(convergence_map, trajectories, costs, nash_eqs, T, min_P, max_P)

    fig_path = os.path.join(FIG_DIR, "Fig2_convergence_map.pdf")
    plt.savefig(fig_path, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")
    plt.show()
    print("Finished")
