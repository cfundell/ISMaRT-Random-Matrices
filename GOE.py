import sys
import numpy as np
import matplotlib.pyplot as plt

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
def calculate_mean_and_var(H, N):
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

    return diag_mean, diag_variance, off_diag_mean, off_diag_variance

def calculate_eigval_spacing(H, N):
    eigenvalues = np.linalg.eigvals(H)
    eigenvalues = -np.sort(-eigenvalues)

    spacings = []
    total_spacing = 0.0

    for val in range(N-1):
        spacing = eigenvalues[val] - eigenvalues[val + 1]
        spacings.append(float(spacing))
        total_spacing += spacing

    average_spacing = total_spacing/N

    return average_spacing, spacings


# Initialize numpy random number generator
rng = np.random.default_rng()

# Test
N = 100
H = generate_GOE(N)
average_eigval_spacing, spacings = calculate_eigval_spacing(H,N)
# print(f"Eigenvalue spacings:\n{spacings}")
print(f"\nAverage eigenvalue spacing: {average_eigval_spacing}")


diag_mean, diag_variance, off_diag_mean, off_diag_variance = calculate_mean_and_var(H, N)

print(f"Mean of diagonal elements: {diag_mean}")
print(f"Variance of diagonal elements: {diag_variance}")
print(f"Mean of off-diagonal elements: {off_diag_mean}")
print(f"Variance of off-diagonal elements: {off_diag_variance}")