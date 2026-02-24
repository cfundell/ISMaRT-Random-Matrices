import sys
import numpy as np
import matplotlib.pyplot as plt

# TEST 
# Generate (symmetric) Gaussian Orthogonal Ensemble of size N 
def generate_GOE(N):
    H = [[0 for col in range(N)] for row in range(N)]

    for i in range(N):
        for j in range(N):
            if j <= i:
                if i == j:
                    rand_num = rng.normal(0.0, 1.0)
                    H[i][j] = rand_num
                else:
                    rand_num = rng.normal(0.0, np.sqrt(0.5)) 
                    H[i][j] = rand_num
                    H[j][i] = rand_num
    return H

# Assumes symmetric matrix
# H is a GOE of size N
# If print_results = True, prints the mean and variance of the diagonal and off-diagonal elements to the terminal
def calculate_mean_and_var(H, N, print_results):
    diag_total = 0.0
    off_diag_total = 0.0

    diag_entries = []
    off_diag_entries = []

    for i in range(N):
        for j in range(N):
            if i == j:
                diag_entries.append(H[i][j])
                diag_total += H[i][j]
            else: 
                off_diag_entries.append(H[i][j])
                off_diag_total += H[i][j]

    diag_mean = diag_total/N
    diag_variance = np.var(diag_entries)
    off_diag_mean = off_diag_total/(N*N - N)
    off_diag_variance = np.var(off_diag_entries)

    if print_results:
        print(f"Mean of diagonal elements: {diag_mean}")
        print(f"Variance of diagonal elements: {diag_variance}")
        print(f"Mean of off-diagonal elements: {off_diag_mean}")
        print(f"Variance of off-diagonal elements: {off_diag_variance}")

    return diag_mean, diag_variance, off_diag_mean, off_diag_variance

# print_results = true: Prints average spacing and its deviation to terminal
# plot_results = true: Plots histograms of the eigenvalues and eigenvalue spacings. Also plots a selected range of eigenvalues to visualize their spacings
def calculate_eigval_spacing(H, N, print_results, plot_results):
    eigenvalues = np.linalg.eigvals(H)
    eigenvalues = -np.sort(-eigenvalues)

    spacings = []
    total_spacing = 0.0

    for val in range(N-1):
        spacing = eigenvalues[val] - eigenvalues[val + 1]
        spacings.append(float(spacing))
        total_spacing += spacing

    average_spacing = total_spacing/N
    spacing_std = np.sqrt(np.var(spacings))

    if print_results:
        print(f"\nAverage eigenvalue spacing: {average_spacing}")
        print(f"Standard deviation of eigenvalue spacing: {spacing_std}")

    if plot_results:
        # Create the histogram probability density for eigenvalues
        plt.hist(
            eigenvalues,
            bins=50,
            range=(-eigenvalues[0], eigenvalues[0]),
            density=True,        # y-axis becomes probability density
            edgecolor="black",   # draw lines around bins
            linewidth=0.5,
            label="Eigenvalue histogram",
            color="#ff9100")
        plt.xlabel("lambda")
        plt.ylabel("Probability density of lambda")
        plt.title(f"Eigenvalue histogram for GOE with N = {N}")
        plt.savefig("eigval_hist.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create histogram probability density for eigenvalue spacings
        plt.hist(
            spacings,
            bins=25,
            range=(0, 0.3),
            density=True,        # y-axis becomes probability density
            edgecolor="black",   # draw lines around bins
            linewidth=0.5,
            label="Eigenvalue histogram",
            color="#ff9100")
        plt.xlabel("|lambda_k - lambda_k+1|")
        plt.ylabel("Probability density")
        plt.title(f"Eigenvalue spacing histogram for GOE with N = {N}")
        plt.savefig("spacings_hist.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Plot eigenvalues to show spacings
        x_range = []
        for i in range(25):
            x_range.append(eigenvalues[500 + i])
        
        y_range = []
        for x in x_range:
            y_range.append(1)

        # Hide tick and label marks of y axis
        fig, ax = plt.subplots()
        ax.set_yticks([]) 
        ax.set_yticklabels([]) 

        plt.plot(x_range, y_range, "bo", markersize=1)
        plt.xlabel("lambda")
        plt.title(f"Eigenvalue spacing histogram for GOE with N = {N} in the range {eigenvalues[500]:.2f},{eigenvalues[524]:.2f}")
        plt.savefig("eigvals.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    return average_spacing, spacings, eigenvalues


# Initialize numpy random number generator
rng = np.random.default_rng()

# Test
N = 1000
H = generate_GOE(N)

calculate_eigval_spacing(H, N, False, True)
