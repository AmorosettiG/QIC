# Assignment 7

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

#######################################################################################################################
# Preliminary definitions : 

# We define Pauli and identity matrices that we need to construct the Hamiltonian : 

sigma_z = np.array([[1, 0], [0, -1]], dtype = complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype = complex)
identity = np.eye(2, dtype = complex)

# We define a function that returns the tensor product of a list of operators :

def tensor_product_operators(operators_list) :
    #  operators_list is a list of 2x2 matrices
    
    result = operators_list[0]
    
    for operator in operators_list[1:] :
        result = np.kron(result, operator)
        
    return result

#######################################################################################################################

# 1) Write a program to compute the matrix representation of the Hamiltonian H for different size N.

def hamiltonian(N, S) :
    # N is the number of spin-1/2 particles
    # S is the strength of the external field (λ)

    H = np.zeros((2**N, 2**N), dtype = complex)
    
    # Filling the matrix with all the external field terms : S * sum sigma_z(i)
    for i in range(N) :
        ops = [identity] * N
        ops[i] = sigma_z
        H += S * tensor_product_operators(ops)
    
    # Adding the interaction term : sum sigma_x(i) * sigma_x(i+1)
    for i in range(N - 1) :
        ops = [identity] * N
        ops[i] = sigma_x
        ops[i + 1] = sigma_x
        H += tensor_product_operators(ops)
    
    return H

#######################################################################################################################

# 2) Diagonalize H for different N = 1, ..., N_max and λ ∈ [0,-3]. 

def diagonalize_H(H) :
    # H is the Hamiltonian matrix to diagonalize
    
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    return eigenvalues, eigenvectors

#######################################################################################################################

# 3) Plot the first k levels as a function of λ for different N and comment on the spectrum.

def plot_spectrum(N_list, lambda_range, k_levels = 10) :
    # N_list is the list of different N values to consider (one subplot for each)
    # lambda_range : values of lambda to consider for the computation (array of values from -3 to 0)
    # k_levels : number of energy levels to plot on each subplot
    
    num_plots = len(N_list)
    
    # Determining the grid layout for subplots depending on N
    ncols = 4  # Adjust as needed
    nrows = (num_plots + ncols - 1) // ncols  # Needed numbers of rows
    
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (15, 5 * nrows))
    axes = axes.flatten()  # Flatten axes for easier iteration
    
    for idx, N in enumerate(N_list) :
        ax = axes[idx]
        eigenvalues_for_N = []
        
        # Computing the eigenvalues for each lambda value 
        for i in lambda_range :
            H = hamiltonian(N, i)
            eigenvalues, _ = diagonalize_H(H)
            eigenvalues_for_N.append(sorted(eigenvalues[:min(k_levels, len(eigenvalues))]))
        
        eigenvalues_for_N = np.array(eigenvalues_for_N)
        num_levels_to_plot = min(k_levels, eigenvalues_for_N.shape[1])
        
        # Ploting the spectrum for this value of N
        for k in range(num_levels_to_plot) :
            ax.plot(lambda_range, eigenvalues_for_N[:, k], label = f'Level {k + 1}')
        
        ax.set_title(f'N = {N}', fontsize = 14)
        ax.set_xlabel(r'$\lambda$', fontsize = 12)
        ax.set_ylabel('Energy', fontsize = 12)
        ax.grid(True)
        ax.legend(fontsize = 7, loc = 'lower right')
    
    # Turning off unused subplots
    for idx in range(len(N_list), len(axes)) :
        axes[idx].axis('off')
    
    # Adjusting the layout to fit correctly all the subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.4, wspace = 0.4)
    plt.savefig(f'plots_N_{N}.pdf')
    plt.show()

#######################################################################################################################

# Parameters for the computations and plots

N_max = 10  # Maximum number of spins to consider
lambda_min, lambda_max = 0, -3
lambda_steps = 50
lambda_values = np.linspace(lambda_min, lambda_max, lambda_steps)
N_list = range(1, N_max + 1)

# Plotting the different spectra for each N 
plot_spectrum(N_list, lambda_values, k_levels = 12)

