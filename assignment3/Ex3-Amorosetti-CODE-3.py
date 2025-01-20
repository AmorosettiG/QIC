# Assignment 3, Exercise 3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Matrix size
N = 1000  

# Number of realizations (to average over)
num_matrices = 100  

# Number of bins for the histogram
num_bins = 1000  

# Setting seed for reproducibility
np.random.seed(12345)  

def hermitian(N) :
    # Generate a random Hermitian matrix of size N
    
    # Random complex matrix of size N, sampled from the standard normal distribution
    cx_matrix = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    
    # Forcing the diagonal to be made of real numbers
    np.fill_diagonal(cx_matrix, np.random.randn(N))
    
    # Keeping only the lower triangle of the complex matrix
    # and adding to itself its conjuguate transpose (excluding the diagonal)
    # So we ensure Hermitian symmetry
    H = np.tril(cx_matrix) + np.tril(cx_matrix, -1).conj().T
    
    return H

def diagonal(N) :
    # Generate a diagonal matrix with random real entries
    
    D = np.diag(np.random.randn(N))
    
    return D 

def normalized_spacings(eigenvalues) :
    # Calculate normalized spacings between sorted eigenvalues
    
    # Spacings: Lambda_i = lambda_{i+1} - lambda_i
    # np.diff() gives the spacings between consecutive eigenvalues
    Lambda = np.diff(eigenvalues)
    
    # Average spacing
    Lambda_avg = np.mean(Lambda)
    
    # Computing the normalized spacings : s_i = Lambda_i / Lambda_avg
    s = Lambda / Lambda_avg
    
    return s

def distribution(matrix_type) :
    # Function to compute P(s) for a given matrix type: 'hermitian' or 'diagonal'
    
    spacings = []

    for _ in range(num_matrices) :
        # Generating the matrix and computing the eigenvalues
        
        if matrix_type == 'hermitian':
            A = hermitian(N)
            
        elif matrix_type == 'diagonal':
            A = diagonal(N)
        
        if matrix_type == 'hermitian' :
            # We diagonalize A to get eigenvalues, that we sort in ascending order
            eigenvalues = np.linalg.eigh(A)[0] 
            # [0] because only eigenvalues are needed, [1] is for eigenvectors
 
        else : 
            eigenvalues = np.linalg.eigvalsh(A)
            # Eigenvalues for the diagonal matrix
                
        # Discarding the first eigenvalue
        eigenvalues = eigenvalues[1:]
        
        # Calculate normalized spacings
        s_i = normalized_spacings(eigenvalues)
        
        spacings.extend(s_i)
        
        # Apply a small threshold to avoid near-zero spacings
        
    # Histogram and normalization to get probability distribution
    counts, bin_edges = np.histogram(spacings, bins=num_bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    return bin_centers, counts

def fit(bin_centers, counts) :
    # Fit P(s) = a * s^alpha * exp( b * s^beta) to the distribution
    
    def Wigner_Dyson(s, a, b, alpha, beta) :
        
        # Function to fit (Wigner-Dyson distribution)
        f = a * (s**alpha) * np.exp(-b * (s**beta))
        
        return f
    
    # Initial guess for a, b, alpha, beta
    initial_params = [1, 1, 1, 1]  
    
    # Set reasonable bounds for each parameter
    bounds = ([0, -5, 0.0, 0.0], [20, 5, 5, 5])  


    # Fit the function to the data
    try :
        # Perform the fit
        params, approx_cov = curve_fit(Wigner_Dyson, bin_centers, counts, p0 = initial_params, bounds = bounds)
        return params
    
    except RuntimeError :
        # In case convergence of the parameters is not reached and the fit fails
        print("Optimal parameters not found, try a different initialization.")
        return None

# P(s) for a random Hermitian matrix
hermitian_bin_centers, hermitian_counts = distribution('hermitian')
hermitian_fit_params = fit(hermitian_bin_centers, hermitian_counts)

# P(s) for a diagonal matrix with real random entries
diagonal_bin_centers, diagonal_counts = distribution('diagonal')
diagonal_fit_params = fit(diagonal_bin_centers, diagonal_counts)

# Priting the fit parameters
if hermitian_fit_params is not None:
    print("Hermitian matrix fit parameters (a, b, alpha, beta) :", hermitian_fit_params)
if diagonal_fit_params is not None:
    print("Diagonal matrix fit parameters (a, b, alpha, beta) :", diagonal_fit_params)

# Plotting results
plt.figure(figsize=(10, 10))

# Hermitian matrix P(s)
plt.bar(hermitian_bin_centers, hermitian_counts, width=hermitian_bin_centers[1] - hermitian_bin_centers[0], 
        alpha=0.6, label='Hermitian Matrix P(s)', color='blue')

if hermitian_fit_params is not None :
    
    plt.plot(hermitian_bin_centers, hermitian_fit_params[0] * hermitian_bin_centers**hermitian_fit_params[2] *
             np.exp(-hermitian_fit_params[1] * hermitian_bin_centers**hermitian_fit_params[3]),
             
             label=f'Fit (Hermitian): a={hermitian_fit_params[0]:.2f}, b={hermitian_fit_params[1]:.2f}, '
                   f'α={hermitian_fit_params[2]:.2f}, β={hermitian_fit_params[3]:.2f}')

# Diagonal matrix P(s)
plt.bar(diagonal_bin_centers, diagonal_counts, width=diagonal_bin_centers[1] - diagonal_bin_centers[0],
        alpha=0.6, label='Diagonal Matrix P(s)', color='orange')

if diagonal_fit_params is not None :
    
    plt.plot(diagonal_bin_centers, diagonal_fit_params[0] * diagonal_bin_centers**diagonal_fit_params[2] *
             np.exp(-diagonal_fit_params[1] * diagonal_bin_centers**diagonal_fit_params[3]),
             
             label=f'Fit (Diagonal): a={diagonal_fit_params[0]:.2f}, b={diagonal_fit_params[1]:.2f}, '
                   f'α={diagonal_fit_params[2]:.2f}, β={diagonal_fit_params[3]:.2f}')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Normalized Spacing $s$', fontsize=16)
plt.ylabel('$P(s)$', fontsize=16)
plt.legend(fontsize=12)
plt.title('Distribution of Normalized Spacings $P(s)$ \nfor Hermitian and Diagonal Matrices', fontsize=16)

# Limiting the x-axis to focus on values up to 5
plt.xlim(0, 5)

plt.savefig('normalized_spacings_hermitian_diag.pdf', bbox_inches='tight')

plt.show()


