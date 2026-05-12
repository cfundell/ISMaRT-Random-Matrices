import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, eigvals


# ============================================================
# 1. Construct Connor's matrix A = T_n + 2D_n
# ============================================================

def make_T(n):
    """
    Construct T_n of size 2n x 2n.

    This is the shift-type matrix used in Connor's A = T_n + 2D_n.
    """
    T = np.zeros((2*n, 2*n), dtype=float)

    # Put identity blocks shifted by 2 columns
    for c in range(2*n - 2):
        T[c, c + 2] = 1.0

    # Wrap-around part
    T[2*n - 2, 0] = 1.0
    T[2*n - 1, 1] = 1.0

    return T


def rotation(theta):
    """
    2x2 rotation matrix.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ], dtype=float)


def make_D(n, theta_scale=1.0):
    """
    Construct D_n of size 2n x 2n using 2x2 rotation blocks.

    theta_k = theta_scale * pi * k / n,  k = 1,...,n.
    """
    D = np.zeros((2*n, 2*n), dtype=float)

    for k in range(1, n + 1):
        theta = theta_scale * np.pi * k / n
        block = rotation(theta)

        i = 2 * (k - 1)
        D[i:i+2, i:i+2] = block

    return D


def make_A(n, theta_scale=1.0):
    """
    Connor-style matrix:
        A = T_n + 2D_n.
    """
    return make_T(n) + 2 * make_D(n, theta_scale)


# ============================================================
# 2. Compute pseudospectrum data
# ============================================================

def pseudospectrum_data(A, grid_size=180, padding=1.0, fixed_radius=None):
    """
    Compute log10(s_min(zI - A)) on a square grid.
    """
    A = np.asarray(A, dtype=complex)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    m = A.shape[0]
    I = np.eye(m, dtype=complex)
    lam = eigvals(A)

    if fixed_radius is None:
        r = max(1.0, np.max(np.abs(lam))) + padding
    else:
        r = fixed_radius

    x = np.linspace(-r, r, grid_size)
    y = np.linspace(-r, r, grid_size)

    Smin = np.zeros((grid_size, grid_size))

    for j, im in enumerate(y):
        for i, re in enumerate(x):
            z = re + 1j * im
            singular_values = svd(z * I - A, compute_uv=False)
            Smin[j, i] = singular_values[-1]

    Smin = np.maximum(Smin, 1e-16)
    Z = np.log10(Smin)

    X, Y = np.meshgrid(x, y)

    return X, Y, Z, lam, r


# ============================================================
# 3. Plot one pseudospectrum
# ============================================================

def plot_one_pseudospectrum(
    A,
    title="Pseudospectrum",
    grid_size=180,
    padding=1.0,
    levels=(-3, -2.5, -2, -1.5, -1, -0.5),
    fixed_radius=None,
    save_path=None
):
    X, Y, Z, lam, r = pseudospectrum_data(
        A,
        grid_size=grid_size,
        padding=padding,
        fixed_radius=fixed_radius
    )

    plt.figure(figsize=(7, 6))

    contour = plt.contour(X, Y, Z, levels=levels)
    plt.clabel(contour, inline=True, fontsize=8)

    plt.scatter(lam.real, lam.imag, color="black", s=15, label="eigenvalues")

    plt.axhline(0, linewidth=0.5)
    plt.axvline(0, linewidth=0.5)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel(r"$\operatorname{Re} z$")
    plt.ylabel(r"$\operatorname{Im} z$")
    plt.title(title)
    plt.legend(loc="upper right")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================
# 4. Batch plot: fixed n, compare theta scales
# ============================================================

def compare_theta_scales(
    n=10,
    theta_scales=(2.0, 1.0, 0.75, 0.5, 0.25),
    grid_size=160,
    levels=(-3, -2.5, -2, -1.5, -1, -0.5),
    fixed_radius=3.0,
    save_path=None
):
    """
    Fixed n, compare different theta choices:
        theta_k = theta_scale * pi*k/n.
    """
    num = len(theta_scales)
    fig, axes = plt.subplots(1, num, figsize=(5*num, 4.5))

    if num == 1:
        axes = [axes]

    for ax, scale in zip(axes, theta_scales):
        A = make_A(n, theta_scale=scale)
        X, Y, Z, lam, r = pseudospectrum_data(
            A,
            grid_size=grid_size,
            fixed_radius=fixed_radius
        )

        contour = ax.contour(X, Y, Z, levels=levels)
        ax.clabel(contour, inline=True, fontsize=7)

        ax.scatter(lam.real, lam.imag, color="black", s=10)

        ax.axhline(0, linewidth=0.5)
        ax.axvline(0, linewidth=0.5)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(rf"$n={n}$, $\theta_k={scale}\pi k/n$")
        ax.set_xlabel(r"$\operatorname{Re} z$")
        ax.set_ylabel(r"$\operatorname{Im} z$")

    fig.suptitle(r"Fixed $n$, comparing different $\theta$ choices", fontsize=16)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================
# 5. Batch plot: fixed theta scale, compare n
# ============================================================

def compare_n_values(
    n_values=(4, 6, 8, 10, 12),
    theta_scale=1.0,
    grid_size=150,
    levels=(-3, -2.5, -2, -1.5, -1, -0.5),
    fixed_radius=3.0,
    save_path=None
):
    """
    Fixed theta scale, compare different n.
    Note: matrix size is 2n x 2n.
    """
    num = len(n_values)
    fig, axes = plt.subplots(1, num, figsize=(5*num, 4.5))

    if num == 1:
        axes = [axes]

    for ax, n in zip(axes, n_values):
        A = make_A(n, theta_scale=theta_scale)
        X, Y, Z, lam, r = pseudospectrum_data(
            A,
            grid_size=grid_size,
            fixed_radius=fixed_radius
        )

        contour = ax.contour(X, Y, Z, levels=levels)
        ax.clabel(contour, inline=True, fontsize=7)

        ax.scatter(lam.real, lam.imag, color="black", s=10)

        ax.axhline(0, linewidth=0.5)
        ax.axvline(0, linewidth=0.5)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(rf"$2n={2*n}$, $\theta_k={theta_scale}\pi k/n$")
        ax.set_xlabel(r"$\operatorname{Re} z$")
        ax.set_ylabel(r"$\operatorname{Im} z$")

    fig.suptitle(r"Fixed $\theta$ choice, comparing different $n$", fontsize=16)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================
# 6. Optional: approximate pseudospectrum area
# ============================================================

def approximate_pseudospectrum_area(
    A,
    epsilon=1e-1,
    grid_size=220,
    fixed_radius=3.0
):
    """
    Approximate area of {z : s_min(zI - A) < epsilon}
    using grid counting.
    """
    X, Y, Z, lam, r = pseudospectrum_data(
        A,
        grid_size=grid_size,
        fixed_radius=fixed_radius
    )

    Smin = 10 ** Z
    mask = Smin < epsilon

    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    area = np.sum(mask) * dx * dy

    return area


def table_area_by_n(
    n_values=(4, 6, 8, 10, 12),
    theta_scale=1.0,
    epsilon=1e-1,
    grid_size=180,
    fixed_radius=3.0
):
    """
    Print approximate pseudospectrum area for different n.
    """
    print(f"theta_scale = {theta_scale}")
    print(f"epsilon = {epsilon}")
    print("-" * 40)
    print(" n     2n     approx area")
    print("-" * 40)

    for n in n_values:
        A = make_A(n, theta_scale=theta_scale)
        area = approximate_pseudospectrum_area(
            A,
            epsilon=epsilon,
            grid_size=grid_size,
            fixed_radius=fixed_radius
        )
        print(f"{n:2d}    {2*n:3d}    {area:.6f}")


# ============================================================
# 7. Run examples
# ============================================================

if __name__ == "__main__":

    # Example A: one single plot
    n = 10
    theta_scale = 1.0
    A = make_A(n, theta_scale=theta_scale)

    plot_one_pseudospectrum(
        A,
        title=rf"Pseudospectrum of $A=T_n+2D_n$, $n={n}$, $\theta_k=\pi k/n$",
        grid_size=180,
        fixed_radius=3.0,
        levels=(-3, -2.5, -2, -1.5, -1, -0.5),
        save_path="single_pseudospectrum.png"
    )

    # Example B: fixed n, compare theta choices
    compare_theta_scales(
        n=10,
        theta_scales=(2.0, 1.0, 0.75, 0.5, 0.25),
        grid_size=150,
        fixed_radius=3.0,
        levels=(-3, -2.5, -2, -1.5, -1, -0.5),
        save_path="compare_theta_scales.png"
    )

    # Example C: fixed theta, compare n
    compare_n_values(
        n_values=(4, 6, 8, 10, 12),
        theta_scale=1.0,
        grid_size=140,
        fixed_radius=3.0,
        levels=(-3, -2.5, -2, -1.5, -1, -0.5),
        save_path="compare_n_values.png"
    )

    # Example D: approximate area table
    table_area_by_n(
        n_values=(4, 6, 8, 10, 12),
        theta_scale=1.0,
        epsilon=1e-1,
        grid_size=160,
        fixed_radius=3.0
    )
