import sys
import numpy as np
import matplotlib.pyplot as plt

# Generate Ginibre ensemble with x_ij ~ N(0, 1)
def generate_Ginibre(N):
    A = [[0 for col in range(N)] for row in range(N)]

    for i in range(N):
        for j in range(N):
            rand_num = rng.normal(0.0, 1.0)
            A[i][j] = rand_num
    return A

# Calculates eigenvalues for Ginibre ensemble of size N. Returns array of eigenvalues.
# If plot_results is True, plots the eigenvalues as a scatterplot in the complex plane
#   and plots histograms for the real and imaginary parts of the eigenvalues
def calculate_eigvals(A, N, plot_results):
    eigenvalues = np.linalg.eigvals(A)
    # sort eigenvalues from largest to smallest
    eigenvalues = -np.sort(-eigenvalues)

    if plot_results:
        # Create scatter plot of eigenvalues
        reals = eigenvalues.real
        imaginaries = eigenvalues.imag

        plt.plot(reals, imaginaries, linestyle="none", color='blue', marker='o', markersize=1)
        plt.xlabel('Real Axis')
        plt.ylabel('Imaginary Axis')
        plt.title(f'Eigenvalues of Ginibre ensemble with N = {N}')

        #Ensure equal aspect ratio for a proper representation of the complex plane
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.savefig("Gini_eigvals.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create the histogram probability density for real part of eigenvalues
        plt.hist(
            reals,
            bins=25,
            range=(-reals[0], reals[0]),
            density=True,        # y-axis becomes probability density
            edgecolor="black",   # draw lines around bins
            linewidth=0.5,
            label="Eigenvalue histogram",
            color="#ff9100")
        plt.xlabel("Re(lambda)")
        plt.ylabel("Probability density of Re(lambda)")
        plt.title(f"Real part histogram for Ginibre ensemble with N = {N}")
        plt.savefig("Gini_eigval_hist_reals.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create the histogram probability density for imaginary part ofeigenvalues
        plt.hist(
            imaginaries,
            bins=25,
            range=(-reals[0], reals[0]),
            density=True,        # y-axis becomes probability density
            edgecolor="black",   # draw lines around bins
            linewidth=0.5,
            label="Eigenvalue histogram",
            color="#ff9100")
        plt.xlabel("Im(lambda)")
        plt.ylabel("Probability density of Im(lambda)")
        plt.title(f"Imagininary part histogram for Ginibre ensemble with N = {N}")
        plt.savefig("Gini_eigval_hist_imagi.png", dpi=300, bbox_inches='tight')
        plt.close()

    return eigenvalues


# Initialize numpy random number generator
rng = np.random.default_rng()

# Test
N = 100
A = generate_Ginibre(N)

calculate_eigvals(A, N, True)