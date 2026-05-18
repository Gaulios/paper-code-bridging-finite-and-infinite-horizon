import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.linalg import solve_discrete_are
import pickle
import math

class GameData:
    def __init__(self, n, d, N, A, B, Q, R, P):
        self.n = n
        self.d = d
        self.N = N
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.P = P

    def to_dict(self):
        return {
            'A': self.A.tolist(),
            'B': self.B.tolist(),
            'Q': self.Q.tolist(),
            'R': self.R.tolist(),
            'P': self.P.tolist()
        }

def compute_riccati_update(game, K, P_next):
    """
    Apply one step of the coupled Riccati recursion: given K and P_{t+1}, return P_t.
    A: (n, n), B: (N, n, d), Q: (N, n, n), R: (N, d, d),
    K: (N, d, n), P_next: (N, n, n)
    """
    n, N = game.n, game.N
    Q, R = game.Q, game.R
    A_closed = compute_closed_loop_matrix(game, K)
    P = np.zeros((N, n, n))
    for i in range(N):
        P[i] = Q[i] + K[i].T @ R[i] @ K[i] + A_closed.T @ P_next[i] @ A_closed
    return P

def check_stabilizability(A, B, N):
    """
    Check whether the system (A, B) is jointly stabilizable.
    A: (n, n), B: (N, n, d), N: number of players.
    Returns True if the combined input matrix has full column rank in the
    controllability matrix.
    """
    n = A.shape[0]
    d = B.shape[2]
    B_combined = np.zeros((n, N * d))
    for i in range(N):
        B_combined[:, i * d:(i + 1) * d] = B[i]
    controllability_matrix = np.zeros((n, n * N * d))
    for i in range(n):
        controllability_matrix[:, i * (N * d):(i + 1) * (N * d)] = np.linalg.matrix_power(A, i) @ B_combined
    return np.linalg.matrix_rank(controllability_matrix) == n

def compute_closed_loop_matrix(game, K):
    """
    Compute the joint closed-loop matrix A - sum_i B_i K_i.
    K: (N, d, n)
    """
    A_closed = game.A.astype(float).copy()
    for i in range(game.N):
        A_closed -= game.B[i] @ K[i]
    return A_closed

def compute_gradient_descent(game, K_init, eta, num_iterations):
    """
    Policy gradient descent toward a Nash equilibrium.
    K_init: (N, d, n) initial policy.
    eta: step size.
    Returns K, P, K_best, grad.
    """
    n, d, N = game.n, game.d, game.N
    B, Q, R = game.B, game.Q, game.R

    K = np.zeros((num_iterations, N, d, n))
    K[0] = K_init
    P = np.zeros((num_iterations - 1, N, n, n))
    grad = np.ones((num_iterations - 1, N, d, n))
    min_grad_norm = 1000
    K_best = K[0]

    if np.max(np.abs(np.linalg.eigvals(compute_closed_loop_matrix(game, K[0])))) >= 1:
        print("Initial closed-loop system is unstable, stopping early.")
        return K, P, None, grad

    for i in range(1, num_iterations):
        A_closed = compute_closed_loop_matrix(game, K[i - 1])
        if np.max(np.abs(np.linalg.eigvals(A_closed))) >= 1:
            print(f"Closed-loop system is unstable at iteration {i - 1}, stopping early.")
            return K[:i - 1], P[:i - 2], K_best, grad[:i - 1]

        fro_norm_grad = 0
        for j in range(N):
            P[i - 1, j] = solve_discrete_lyapunov(
                A_closed.T, Q[j] + K[i - 1, j].T @ R[j] @ K[i - 1, j]
            )
            grad[i - 1, j] = (
                (R[j] + B[j].T @ P[i - 1, j] @ B[j]) @ K[i - 1, j]
                - B[j].T @ P[i - 1, j] @ (A_closed + B[j] @ K[i - 1, j])
            )
            K[i, j] = K[i - 1, j] - eta * grad[i - 1, j]
            fro_norm_grad += np.linalg.norm(grad[i - 1, j], 'fro')

        if fro_norm_grad < N * 1e-4:
            print(f"Convergence reached at iteration {i}")
            return K[:i], P[:i - 1], K[i], grad[:i]

        if fro_norm_grad > min_grad_norm:
            K_best = K[i]

        if (10 * i) % num_iterations == 0:
            print(f"Gradient descent: iteration {i}")

    print("Reached maximum iterations without convergence. Last norm of the gradient:",
          np.linalg.norm(grad[-1]))
    return K, P, K_best, grad

def run_backward_riccati(T, game, P_terminal):
    """
    Run the backward coupled Riccati recursion over a horizon T.
    P_terminal: (N, n, n) terminal cost matrices.
    Returns K of shape (T, N, d, n) and Pt of shape (T+1, N, n, n).
    """
    n, d, N = game.n, game.d, game.N

    K = np.zeros((T, N, d, n))
    Pt = np.zeros((T + 1, N, n, n))
    Pt[T] = P_terminal

    for t in range(T):
        current_time = T - t
        M_inv = compute_coupling_matrix_inverse(game, Pt[current_time])
        if M_inv is None:
            return None
        rhs = build_rhs_vector(game, Pt[current_time])
        Kt = M_inv @ rhs
        for i in range(N):
            K[current_time - 1, i] = Kt[i * d:(i + 1) * d, :]
        Pt[current_time - 1] = compute_riccati_update(game, K[current_time - 1], Pt[current_time])

    return K, Pt

def build_rhs_vector(game, P):
    """
    Build the stacked right-hand-side vector [B_i^T P_i A]_{i=1..N}.
    Output shape: (N*d, n)
    """
    n, d, N = game.n, game.d, game.N
    A, B = game.A, game.B
    rhs = np.zeros((N * d, n))
    for i in range(N):
        rhs[i * d:(i + 1) * d, :] = B[i].T @ P[i] @ A
    return rhs

def detect_convergence_type(K, T, N):
    """
    Determine the asymptotic behavior of the backward Riccati recursion.
    Returns:
      1   if K converges to a fixed point,
      L>1 if K converges to a cycle of length L,
      -1  if no convergence is detected.
    K has shape (T, N, d, n); the tail of the sequence is used for detection.
    """
    W = 50
    tol = 1e-6
    tail = K[:W]

    if np.max(np.sqrt(np.sum((tail - tail[0]) ** 2, axis=(1, 2, 3)))) < tol:
        return 1

    for L in range(2, min(T // 5, W // 2) + 1):
        periodic = True
        for t in range(W - L):
            if np.linalg.norm(tail[t] - tail[t + L]) > tol:
                periodic = False
                break
        if periodic:
            return L

    return -1

def find_stabilizing_policy(game):
    """
    Compute a jointly stabilizing policy K via the single-agent DARE.
    Returns K of shape (N, d, n).
    """
    n, d, N = game.n, game.d, game.N
    A, B = game.A, game.B
    Q_lqr = np.eye(n)
    R_lqr = np.eye(N * d)
    B_combined = np.zeros((n, N * d))
    for i in range(N):
        B_combined[:, i * d:(i + 1) * d] = B[i]
    P_dare = solve_discrete_are(A, B_combined, Q_lqr, R_lqr)
    K_total = np.linalg.inv(R_lqr + B_combined.T @ P_dare @ B_combined) @ B_combined.T @ P_dare @ A
    K = np.zeros((N, d, n))
    for i in range(N):
        K[i] = K_total[i * d:(i + 1) * d, :]
    return K

def generate_random_game(n, d, N):
    """
    Sample a random LQ game with positive-definite cost matrices.
    Resamples B until the system is stabilizable.
    """
    A = np.random.randn(n, n)
    stab = False
    while not stab:
        B = np.random.randn(N, n, d)
        stab = check_stabilizability(A, B, N)
    Q = np.zeros((N, n, n))
    R = np.zeros((N, d, d))
    P = np.zeros((N, n, n))
    for i in range(N):
        Qtemp = np.random.randn(n, n)
        Q[i] = Qtemp.T @ Qtemp + 0.01 * np.eye(n)
        Rtemp = np.random.randn(d, d)
        R[i] = Rtemp.T @ Rtemp + 0.01 * np.eye(d)
        Ptemp = np.random.randn(n, n)
        P[i] = Ptemp.T @ Ptemp + 0.01 * np.eye(n)
    return GameData(n, d, N, A, B, Q, R, P)

def compute_coupling_matrix_inverse(game, P):
    """
    Invert the block coupling matrix M whose (i,j) block is
    B_i^T P_i B_j  (plus R_i on the diagonal).
    B: (N, n, d), R: (N, d, d), P: (N, n, n).
    Returns M^{-1} of shape (N*d, N*d), or None if M is singular.
    """
    d, N = game.d, game.N
    B, R = game.B, game.R
    M = np.zeros((N * d, N * d))
    for i in range(N):
        for j in range(N):
            if i == j:
                M[i * d:(i + 1) * d, i * d:(i + 1) * d] = B[i].T @ P[i] @ B[i] + R[i]
            else:
                M[i * d:(i + 1) * d, j * d:(j + 1) * d] = B[i].T @ P[i] @ B[j]
    if np.abs(np.linalg.det(M)) == 0:
        with open('gameWithZeroDeterminant.pkl', 'wb') as f:
            pickle.dump((B, R, P), f)
        print("The determinant of the coupling matrix M is zero.")
        return None
    return np.linalg.inv(M)


### FIXED-POINT ANALYSIS ###################################################

def one_step_map(P1, P2, game):
    """One backward Riccati step (P1, P2) -> (P1', P2') for the scalar case."""
    P = np.zeros((game.N, 1, 1))
    P[0, 0, 0] = P1
    P[1, 0, 0] = P2
    M_inv = compute_coupling_matrix_inverse(game, P)
    rhs = build_rhs_vector(game, P)
    Kt = M_inv @ rhs
    K_split = np.zeros((game.N, 1, 1))
    for i in range(game.N):
        K_split[i] = Kt[i:i + 1, :]
    P_new = compute_riccati_update(game, K_split, P)
    return P_new[0, 0, 0], P_new[1, 0, 0]

def jacobian_at_fixed_point(P1_star, P2_star, game, eps=1e-6):
    """Numerically compute the Jacobian of the Riccati map at a fixed point."""
    f0 = one_step_map(P1_star, P2_star, game)
    f1 = one_step_map(P1_star + eps, P2_star, game)
    f2 = one_step_map(P1_star, P2_star + eps, game)
    return np.array([
        [(f1[0] - f0[0]) / eps, (f2[0] - f0[0]) / eps],
        [(f1[1] - f0[1]) / eps, (f2[1] - f0[1]) / eps]
    ])

def classify_fixed_point(eigenvalues):
    """Classify a fixed point of the Riccati map from the Jacobian eigenvalues."""
    mags = np.abs(eigenvalues)
    if np.all(mags < 1):
        return "stable node (attractor)"
    elif np.all(mags > 1):
        return "unstable node (repeller)"
    else:
        return "saddle point"


### PLOT FUNCTIONS ##########################################################
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_convergence_map(convergence_map, trajectories, costs, nash_eqs, T, vmin, vmax):
    """
    Plot the convergence map of the finite-horizon Riccati recursion (scalar case).
    Each pixel indicates which infinite-horizon NE the recursion converges to,
    starting from terminal cost (P1, P2). Overlaid trajectories show sample paths.
    """
    neg_cmap = plt.get_cmap('summer', 128)
    pos_cmap = plt.get_cmap('hot_r', 128)
    newcolors = np.vstack((neg_cmap(np.linspace(0, 1, 128)), pos_cmap(np.linspace(0, 1, 128))))
    combined_cmap = ListedColormap(newcolors, name='negpos')

    v_min_data = np.min(convergence_map)
    v_max_data = np.max(convergence_map)
    bounds = np.linspace(v_min_data, v_max_data, 257)
    norm = BoundaryNorm(bounds, combined_cmap.N)

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.figure(figsize=(8, 8))
    plt.imshow(convergence_map, extent=[vmin, vmax, vmin, vmax], origin='lower',
               cmap=combined_cmap, norm=norm, interpolation='none', aspect='auto')

    for traj in trajectories:
        for t in range(T):
            plt.plot(traj[t, 0], traj[t, 1], 'o', markersize=5, color="black")
        plt.plot(traj[T, 0], traj[T, 1], 's', markersize=7, color="blue")
        plt.quiver(
            traj[1:, 0], traj[1:, 1],
            traj[:-1, 0] - traj[1:, 0], traj[:-1, 1] - traj[1:, 1],
            angles='xy', scale_units='xy', scale=1, color="gray",
            width=0.0015, headwidth=6, headlength=15, headaxislength=10
        )

    NEcolor = plt.get_cmap('Dark2')
    for idx in range(nash_eqs.shape[0]):
        plt.plot(costs[idx, 0], costs[idx, 1], '*', markersize=15,
                 color=NEcolor(5 + idx), markeredgecolor='black', label=f'NE {idx + 1}')

    plt.xlabel('$P^1$', fontsize=15)
    plt.ylabel('$P^2$', fontsize=15)
    plt.legend(loc='upper right', fontsize=15)

    # Inset zoom around the saddle-point NE
    zoom_margin = 2
    rect_main = plt.Rectangle(
        (costs[1, 0] - zoom_margin, costs[1, 1] - zoom_margin),
        2 * zoom_margin, 2 * zoom_margin,
        edgecolor='black', facecolor='none', linestyle='-'
    )
    plt.gca().add_patch(rect_main)

    ax_inset = inset_axes(plt.gca(), width="30%", height="30%", loc='lower right', borderpad=2)
    ax_inset.imshow(convergence_map, extent=[vmin, vmax, vmin, vmax], origin='lower',
                    cmap=combined_cmap, norm=norm, interpolation='none', aspect='auto')

    for traj in trajectories:
        for t in range(T):
            ax_inset.plot(traj[t, 0], traj[t, 1], 'o', markersize=3, color="black")
        ax_inset.plot(traj[T, 0], traj[T, 1], 's', markersize=5, color="black")
        ax_inset.quiver(
            traj[1:, 0], traj[1:, 1],
            traj[:-1, 0] - traj[1:, 0], traj[:-1, 1] - traj[1:, 1],
            angles='xy', scale_units='xy', scale=1, color="black",
            width=0.002, headwidth=6, headlength=15, headaxislength=10
        )

    for idx in range(nash_eqs.shape[0]):
        ax_inset.plot(costs[idx, 0], costs[idx, 1], '*', markersize=15,
                      color=NEcolor(5 + idx), markeredgecolor='black')

    zoom_area = [
        costs[1, 0] - zoom_margin, costs[1, 0] + zoom_margin,
        costs[1, 1] - zoom_margin, costs[1, 1] + zoom_margin
    ]
    ax_inset.set_xlim(zoom_area[0], zoom_area[1])
    ax_inset.set_ylim(zoom_area[2], zoom_area[3])
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    rect_zoom = plt.Rectangle(
        (zoom_area[0], zoom_area[2]),
        zoom_area[1] - zoom_area[0], zoom_area[3] - zoom_area[2],
        edgecolor='black', facecolor='none', linestyle='--'
    )
    plt.gca().add_patch(rect_zoom)

    plt.grid(False)
    plt.tight_layout()


def plot_convergence_heatmaps(static, periodic, not_conv, min_n, min_N, max_n, max_N):
    """
    Plot three side-by-side heatmaps showing, for each (n, N) pair, the
    percentage of random games converging to a static NE, a periodic NE,
    or failing to converge.
    """
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=True)

    max_conv = np.max(static)
    min_conv = np.min(static)
    max_loop = np.max(periodic)
    max_not  = np.max(not_conv)

    im0 = axes[0].imshow(static, cmap='summer_r', interpolation='nearest',
                         vmin=min_conv, vmax=max_conv, origin='lower')
    axes[0].set_title('Convergence to\nStatic Nash Equilibrium', fontsize=15)
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label('Percentage', fontsize=15)
    cbar0.set_ticks(np.linspace(min_conv, max_conv, 11))
    cbar0.set_ticklabels([f"{int(100 * v)}%" for v in np.linspace(min_conv, max_conv, 11)])

    im1 = axes[1].imshow(periodic, cmap='Wistia', interpolation='nearest',
                         vmin=0, vmax=max_loop, origin='lower')
    axes[1].set_title('Convergence to\nPeriodic Nash equilibrium', fontsize=15)
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label('Percentage', fontsize=15)
    cbar1.set_ticks(np.linspace(0, max_loop, 8))
    cbar1.set_ticklabels([f"{int(100 * v)}%" for v in np.linspace(0, max_loop, 8)])

    im2 = axes[2].imshow(not_conv, cmap='autumn_r', interpolation='nearest',
                         vmin=0, vmax=max_not, origin='lower')
    axes[2].set_title('Failure to Converge', fontsize=15)
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, shrink=0.6)
    cbar2.set_label('Percentage', fontsize=15)
    cbar2.set_ticks(np.linspace(0, max_not, 11))
    cbar2.set_ticklabels([f"{int(100 * v)}%" for v in np.linspace(0, max_not, 11)])

    for ax in axes:
        ax.set_xlabel('$N$', fontsize=15)
        ax.set_ylabel('$n$', fontsize=15)
        ax.set_xticks(np.arange(max_N + 1 - min_N))
        ax.set_xticklabels(np.arange(min_N, max_N + 1))
        ax.set_yticks(np.arange(max_n + 1 - min_n))
        ax.set_yticklabels(np.arange(min_n, max_n + 1))


def plot_policy_evolution(K, K_fixed, N, zoom, zoom_size=0.02):
    """
    Scatter-plot the trajectory of each agent's policy K[:,i,0,:] in the
    (K[1,1], K[1,2]) plane, colored by time step.  If zoom=True, an inset
    magnifies the region around the fixed point K_fixed.
    """
    T = K.shape[0]
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig, axes = plt.subplots(1, N, figsize=(6 * N, 5), squeeze=False)
    axes = axes[0]

    for i in range(N):
        ax = axes[i]
        x = K[:, i, 0, 0]
        y = K[:, i, 0, 1]
        scatter = ax.scatter(x, y, c=np.arange(T), cmap='viridis', s=150, vmin=0, vmax=T - 1)

        if zoom:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            width  = zoom_size * (xlim[1] - xlim[0])
            height = zoom_size * (ylim[1] - ylim[0])
            rect = plt.Rectangle(
                (K_fixed[i, 0, 0] - width / 2, K_fixed[i, 0, 1] - height / 2),
                width, height,
                linewidth=1.5, edgecolor='grey', facecolor='none', linestyle='--', zorder=4
            )
            ax.add_patch(rect)
            axins = inset_axes(ax, width="40%", height="40%", loc="lower left", borderpad=2)
            axins.scatter(x, y, c=np.arange(T), cmap='viridis', s=150, vmin=0, vmax=T - 1)
            axins.scatter(K_fixed[i, 0, 0], K_fixed[i, 0, 1],
                          color='red', marker='*', edgecolor='black', s=300, zorder=5)
            axins.set_xlim(K_fixed[i, 0, 0] - width / 2, K_fixed[i, 0, 0] + width / 2)
            axins.set_ylim(K_fixed[i, 0, 1] - height / 2, K_fixed[i, 0, 1] + height / 2)
            axins.set_xticks([])
            axins.set_yticks([])

        ax.set_xlabel(f'$K^{i + 1}_t[1,1]$', fontsize=15)
        ax.set_ylabel(f'$K^{i + 1}_t[1,2]$', fontsize=15)
        ax.grid()

    plt.tight_layout(pad=0.5)


def plot_periodic_equilibrium(game, K, P, L, N):
    """
    Visualize a periodic Nash equilibrium of period L.
    Top panel: Frobenius-norm distance ||P^i_t - P^i_0||_F over time.
    Bottom panel: spectral radius of the closed-loop matrix A^cl_t over time.
    """
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    num = 4 * L + 1
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    norm_diff = np.zeros((num, N))
    for l in range(num):
        for i in range(N):
            norm_diff[l, i] = np.linalg.norm(P[l][i] - P[0][i], ord='fro')

    x = np.arange(num)
    for i in range(N):
        axs[0].plot(x, norm_diff[:, i], marker='o', markersize=12, linewidth=3,
                    linestyle='dotted', label=f'Agent {i + 1}')
    axs[0].set_yscale('symlog', linthresh=1e-1)
    axs[0].set_xlim(0, num - 1)
    axs[0].set_ylabel(r'$\|P^i_{t}-P^i_0\|_F$', fontsize=15)
    axs[0].legend(fontsize=15, loc='upper right')
    axs[0].grid()

    spectral_radii = np.zeros((num, 1))
    for l in range(num):
        C = game.A.copy()
        for i in range(N):
            C -= game.B[i] @ K[l][i]
        spectral_radii[l] = np.max(np.abs(np.linalg.eigvals(C)))

    spectral_product = np.prod(spectral_radii[:L])
    print(f"Product of spectral radii over one period: {spectral_product:.6f}")

    axs[1].plot(x, spectral_radii, marker='o', markersize=12, linewidth=3,
                linestyle='dotted', label='Spectral Radius')
    axs[1].axhline(y=1, color='r', linewidth=5, linestyle='-', label='Stability Threshold')
    axs[1].set_yscale('log')
    axs[1].set_xlim(0, num - 1)
    axs[1].set_xlabel('t', fontsize=15)
    axs[1].set_ylabel(r'$\rho \left(A^{cl}_t \right)$', fontsize=15)
    axs[1].legend(fontsize=15, loc='lower right')
    axs[1].grid()

    plt.tight_layout()
