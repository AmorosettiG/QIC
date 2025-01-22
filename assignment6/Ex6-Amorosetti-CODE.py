# Assignment 6

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time

from concurrent.futures import ThreadPoolExecutor, as_completed # To speed up the runtimes measurements using parallel processing
from tqdm import tqdm                                           # To print progress of the loop in real time

# The code is structured following the differents tasks asked in the assignment instructions
################################################################################

# a) Describing the composite system in the case of N-body non interacting, separable pure states : 

def separable_state(coeffs, dimensions) :
    
    # coeffs : list of 1D numpy arrays of the coefficients of each subsystem's wavefunction
    # dimensions : list of dimensions (for each subsystem's Hilbert space)
 
    # Ensuring that each subsystem has an associated 'coeffs' vector before continuing
    assert len(coeffs) == len(dimensions)
    
    for i, dim in enumerate(dimensions) :
        # Ensuring that the length of each 'coeffs' vector is matching the dimension for this subsystem 
        assert len(coeffs[i]) == dim

    state = coeffs[0]  # Initialization
    
    for coeff in coeffs[1:] :   
        state = np.kron(state, coeff) # Kronecker product to compute the tensor product
        
    return state

################################################################################

# b) Describing the composite system in the case of a general N-body pure wave function Ψ ∈ ℋ^D^N

def general_state(coeffs) :
    # coeffs is a 1D np array of size D^N (Hilbert space dimension)
    
    # Computing the norm
    norm = np.linalg.norm(coeffs)
    
    # Normalizing the wavefunction
    state = coeffs / norm
    
    return state 

################################################################################

# c) Efficiency comparison

# Function to measure the runtimes when creating a state, depending on the method (functions implemented above), the dimension and number of subsystems

def runtimes(dim, num_subsystems, method) :

    if method == "separable" :
        
        coeffs = [np.random.rand(dim) + 1j * np.random.rand(dim) for _ in range(num_subsystems)]
        
        for coeff in coeffs :
            coeff /= np.linalg.norm(coeff)
            
        start_time = time.time()
        separable_state(coeffs, [dim] * num_subsystems)
        
        return time.time() - start_time
    
    elif method == "general" :
        
        full_dim = dim ** num_subsystems
        coeffs = np.random.rand(full_dim) + 1j * np.random.rand(full_dim)
        
        start_time = time.time()
        general_state(coeffs)
        
        return time.time() - start_time
    
    else :
        raise ValueError("Error, choose 'separable' or 'general' as a method")
    
#-------------------------------------------------------------------------------

# Function to measure the runtimes using parallel computation (to speed up the computations when the dimension and number of subsystems increase)

def runtimes_parallel(dimensions, num_subsystems, method) :
    
    results = []
    with ThreadPoolExecutor() as executor :
        
        futures = [executor.submit(runtimes, dim, num_subsystems, method) for dim in dimensions]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{method.capitalize()} (N={num_subsystems})") :
            results.append(future.result())
            
    return results

#-------------------------------------------------------------------------------

# Function to be fitted to
def fit_fct(x, a, b) :
    # Power-law
    return a * x**b 

# Function to plot the runtimes depending on the dimension, for different number of subsystems, for both methods and their fit
def plot_efficiency() :

    # Different dimensions for each number (N) of subsystems
    dimensions_dict = {
        2: np.unique(np.logspace(1, 4.1, num=50, dtype=int)),   # N = 2
        3: np.unique(np.logspace(1, 2.75, num=40, dtype=int)),  # N = 3
        4: np.unique(np.logspace(1, 2.05, num=35, dtype=int)),  # N = 4
        5: np.unique(np.logspace(1, 1.65, num=30, dtype=int)),  # N = 5
    }

    num_subsystems_list = [2, 3, 4, 5]  # list of the numbers of subsystems
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, num_subsystems in enumerate(num_subsystems_list) :
        
        dimensions = dimensions_dict[num_subsystems]
        separable_times = runtimes_parallel(dimensions, num_subsystems, "separable")
        general_times = runtimes_parallel(dimensions, num_subsystems, "general")

        # We decided to use only the second half of the points (that have a linear behavior) for the fit
        midpoint = len(dimensions) // 2
        fit_dimensions = dimensions[midpoint:]
        fit_separable_times = separable_times[midpoint:]
        fit_general_times = general_times[midpoint:]

        # Fitting
        separable_params, _ = curve_fit(fit_fct, fit_dimensions, fit_separable_times)
        general_params, _ = curve_fit(fit_fct, fit_dimensions, fit_general_times)

        # Data to plot from the fits
        separable_fit = fit_fct(dimensions, *separable_params)  
        general_fit = fit_fct(dimensions, *general_params) 


        ax = axes[idx]
        ax.loglog(dimensions, separable_times, label=f"Separable", marker="o", color="blue", linestyle="none", markersize=4)
        ax.loglog(dimensions, general_times, label=f"General", marker="o", color="red", linestyle="none", markersize=4)

        ax.loglog(dimensions, separable_fit, 
          label=fr"Separable fit: ${separable_params[0]:.2e} \cdot D^{{{separable_params[1]:.2f}}}$", 
          linestyle=(0, (3, 1)), color="lightblue")
        ax.loglog(dimensions, general_fit, 
          label=fr"General fit: ${general_params[0]:.2e} \cdot D^{{{general_params[1]:.2f}}}$", 
          linestyle=(0, (3, 1)), color="tomato")


        ax.set_title(f"N = {num_subsystems}", fontsize=14)
        ax.set_xlabel("Dimension (D)", fontsize=12)
        ax.set_ylabel("Time (s)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    fig.suptitle("Runtimes depending on the dimension for separable and general states", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
    plt.savefig('runtimes.pdf')
    plt.show()
    
#-------------------------------------------------------------------------------

# plot_efficiency()      # Commented to run the code without measuring the runtimes neither plotting

################################################################################

# d) N = 2, writing the density matrix of a general pure state Ψ, ρ = |Ψ⟩⟨Ψ|

def density_matrix(state) :
    
    matrix = np.outer(state, state.conj())  # Computing the density matrix with the outer product of ∣Ψ⟩ and ∣Ψ∗⟩
    
    return matrix

################################################################################

# e) Computing the reduced density matrix of either the left or the right system e.g ρ1 = Tr(ρ2)
#    (given a generic density matrix of dimension D^N × D^N)

def reduced_density_matrix(rho, dim, trace_over = "right") :
    
    # rho is the full density matrix of size (D^2, D^2)
    # dim is the dimension of each subsystem
    # trace_over is the subsystem to trace
    
    # Reshaping the full density matrix rho into a 4D array of shape (D, D, D, D)
    rho_4d = rho.reshape(dim, dim, dim, dim)

    if trace_over == "right" :
        # Partial trace over the second (right) subsystem (indices 2 and 3) 
        reduced_rho = np.trace(rho_4d, axis1 = 2, axis2 = 3)
        
    elif trace_over == "left" :
        # Partial trace over the first (left) subsystem
        reduced_rho = np.trace(rho_4d, axis1 = 0, axis2 = 1)

    return reduced_rho 
    # Reduced density matrix, size (D, D)

################################################################################

# f) Testing the functions described before (and all others needed),
#    on two-spin one-half (qubits) with different states

def test() :
    
    # Test of the previous functions with two qubits
    dim = 2
    
    # Separable states
    separable_states = [
        [np.array([1, 0]), np.array([0, 1])],                            # |0⟩ ⊗ |1⟩
        [np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), np.array([1, 0])]   # (|0⟩ + |1⟩) ⊗ |0⟩
    ]
    
    # General states (Bell states)
    general_states = [
        np.array([1, 0, 0, 1]) / np.sqrt(2),  # |Φ+⟩
        np.array([0, 1, 1, 0]) / np.sqrt(2)   # |Ψ+⟩
    ]

    for i, state in enumerate(separable_states + general_states) :
        
        # Separable state, need to compute the Kronecker product
        if isinstance(state, list) :  
            state = separable_state(state, [dim] * len(state))  # Getting the full separable state
        
        rho = density_matrix(state)
        reduced_left = reduced_density_matrix(rho, dim, trace_over = "left")
        reduced_right = reduced_density_matrix(rho, dim, trace_over = "right")
        
        print(f"State {i + 1} : ")
        print(f"Density matrix : \n{rho}\n")
        print(f"Reduced density matrix (left) : \n{reduced_left}\n")
        print(f"Reduced density natrix (right) : \n{reduced_right}\n{'='*40}")

# Running the test function
test()



