# Assignment 3, Exercise 2

import numpy as np
import matplotlib.pyplot as plt

# Size of the considered matrix
N = 5000 

# Setting seed for reproducibility
np.random.seed(12345)  

# Creating a Hermitian matrix using only the lower triangle :

# Random complex matrix of size N, sampled from the standard normal distribution
cx_matrix = np.random.randn(N, N) + 1j * np.random.randn(N, N)

# Forcing the diagonal to be made of real numbers
np.fill_diagonal(cx_matrix, np.random.randn(N))

# Keeping only the lower triangle of the complex matrix
# and adding to itself its conjuguate transpose (excluding the diagonal)
# So we ensure Hermitian symmetry
A = np.tril(cx_matrix) + np.tril(cx_matrix, -1).conj().T  

# We diagonalize A to get eigenvalues, that we sort in ascending order
eigenvalues = np.linalg.eigh(A)[0]  
# [0] because only eigenvalues are needed, [1] is for eigenvectors

# Discarding the first eigenvalue
eigenvalues = eigenvalues[1:]

# Spacings: Lambda_i = lambda_{i+1} - lambda_i
# np.diff() gives the spacings between consecutive eigenvalues
Lambda = np.diff(eigenvalues)  

# Average spacing
Lambda_avg = np.mean(Lambda)

# Computing the normalized spacings : s_i = Lambda_i / Lambda_avg
s = Lambda / Lambda_avg

# Priting the eigenvalues and normalized spacings
print("Eigenvalues:", eigenvalues)
print("Normalized spacings s_i:", s)

# Ploting the histogram of the normalized spacings
plt.figure(figsize=(8, 8))

plt.hist(s, bins=100, density=False, alpha=0.7, color='blue', edgecolor='black')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Normalized spacing $s_i$", fontsize=16)
plt.ylabel("Counts", fontsize=16)
plt.title(f"Histogram of the normalized spacings between the eigenvalues \nof a random Hermitian matrix of size N = {N}", fontsize=18)
plt.savefig('normalized_spacings.pdf', bbox_inches='tight')

plt.show()
