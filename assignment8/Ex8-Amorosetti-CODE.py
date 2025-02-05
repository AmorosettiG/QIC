# Assignment 8

import numpy as np
import matplotlib.pyplot as plt   
import scipy.sparse as sparse   
# We use the sparse package of the SciPy library for more efficient computations, with sparse matrices and their built-in optimized functions

# We define Pauli and identity matrices that we need to construct the Hamiltonian :
# (in the same way as in the previous assignment but with the sparse built-in functions)
 
sigma_x = sparse.csc_matrix([[0, 1], [1, 0]], dtype = complex)
sigma_z = sparse.csc_matrix([[1, 0], [0, -1]], dtype = complex) 
identity = sparse.identity(2, format = "csr", dtype = complex) # Compressed Sparse Row (CSR) Format, to store sparse matrices efficiently by compressing the rows


# We define a function that returns the tensor product of a list of operators :
# (same function as in the previous assignment but with the sparse built-in functions)

def tensor_product_operators(operators_list) :
    #  operators_list is a list of 2x2 matrices
    
    result = operators_list[0]
    
    for operator in operators_list[1:] :
        result = sparse.kron(result, operator, format = "csr")
        
    return result

# We define a function to compute the matrix representation of the Hamiltonian H for different size N
# (same function as in the previous assignment but with the sparse built-in functions)

def hamiltonian(N, S):
    # N is the number of spin-1/2 particles
    # S is the strength of the external field (λ)
    
    H = sparse.csc_matrix((2**N, 2**N), dtype = complex)
    
    # Filling the matrix with all the external field terms : λ * sum sigma_z(i)
    for i in range (N) :
        ops = [identity] * N
        ops[i] = sigma_z
        H += S * sparse.csc_matrix(tensor_product_operators(ops))
        
    # Adding the interaction term : sum sigma_x(i) * sigma_x(i+1)
    for i in range(N - 1) :
        ops = [identity] * N
        ops[i] = sigma_x
        ops[i + 1] = sigma_x
        H += sparse.csc_matrix(tensor_product_operators(ops))
        
    return H 

######################################################################################################################

# We define a function implementing the Real-Space Renormalization Group (RSRG) algorithm

def RSRG(H, N, n, m, iteration_energy = 1e6, convergence = 1e-14, iteration = 0, energy_history = None) :
    
    # The inputs are : 
    
    # H is the Hamiltonian of the considered system
    # N is the number of particles in the considered system
    # n is the Hilbert space dimension of the considered system
    
    # m is the number of lowest eigenvectors to keep
    # iteration_energy is the ground state energy of the previous iteration
    # convergence (τ) is the value to consider for the energy difference between the current and the previous ground state energies
    
    # iteration counts the number of iterations
    # energy_history is used to save the energy found trough all the iterations
    
    if energy_history is None : 
        energy_history = []
    
    #----------------------------------------------------------------------------------------------
    # INITAL : H(N), interaction operators A(0) and B(0)
    
    A = sparse.kron(sparse.identity(2**(n-1), format = "csc"), sigma_x, format = "csc") # Compressed Sparse Column Format (focuses on column-major storage)
    B = sparse.kron(sigma_x, sparse.identity(2**(n-1), format = "csc"), format = "csc")
    
    #----------------------------------------------------------------------------------------------
    # ALGORITHM
 
    # Find the M lowest energy eigenvectors |v⟩ H(N) of and build the projector P = ∑ |v⟩⟨v|
    
    eigen_values, P = sparse.linalg.eigsh(H, 2**m, which = 'SA', return_eigenvectors = True) # ‘SA’ : smallest (algebraic) eigenvalues
    P = sparse.csc_matrix(P)
    
    current_energy_density = eigen_values[0] / N  # Energy for each site
    energy_history.append(current_energy_density) # Saving the current energy
    
    
    # Check the convergence of the energy density ϵ = E0 / N with a threshold criterion  Δn = |ϵ(n) − ϵ(n−1)| < τ : 
    diff = eigen_values[0] - iteration_energy
    
    if abs(diff) < convergence :
        return N, eigen_values[0], iteration + 1, energy_history        
        # It returns the current system size and ground state energy, as well the number of iterations and the energy history
    
    # If it is not converged : update the Hamiltonian and the interaction for the next iteration 

    
    # Creation of a diagonal matrix with the lowest eigenvalues, i.e. the reduced Hamiltonian in the eigenbasis
    H_diagonal = sparse.diags(eigen_values)
    
    # Build the Hamiltonian and state for the system of size 2N from the previous step
    H_reduced = sparse.kron(H_diagonal, sparse.identity(2**m), format = "csc") + sparse.kron(sparse.identity(2**m), H_diagonal, format = "csc")
    
    
    # Truncate the system description (states and operators) using P ; computing H̃ = P†HP
    
    A_new = sparse.csc_matrix(P.conj().T @ A @ P) # Transform interaction operators A and B into the reduced subspace
    B_new = sparse.csc_matrix(P.conj().T @ B @ P)
    
    # Computing the interaction term for the larger system :
    H_interaction = sparse.kron(A_new, B_new, format = "csc") / N
    
    # Combining the reduced Hamiltonian and the interaction term
    H_new = (H_reduced + H_interaction) / 2
    
    # Finally, recursively perform the RSRG algorithm on the new Hamiltonian and update the iterations counter and the energy history
    return RSRG(H = H_new, N = N*2, n = m*2, m = m, 
                iteration_energy = eigen_values[0],
                convergence = convergence, 
                iteration = iteration + 1, 
                energy_history = energy_history) # Returns the (New system size, Ground state energy) tuple after convergence is reached

######################################################################################################################

# We define a function to compute the mean field solution in order to compare it with the one from the real-space RG algorithm
# Theory from : S. Montangero. "Introduction to Tensor Network Methods : Numerical simulations of low-dimensional many-body quantum systems", page 39, equation 4.8

def mean_field(S) :
    
    # If |λ| ≤ 2, we use the quadratic approximation
    if abs(S) <= 2 :
        return -1 - (S**2)/4 
    
    # If not, we return the linear solution
    else :
        return -abs(S)
    
######################################################################################################################

# Parameters for the computations and plots

lambda_min, lambda_max = -3, 3
lambda_steps = 50   
lambda_values = np.linspace(lambda_min, lambda_max, lambda_steps)

# Parameters of the initial system
N = 14      # Number of particles
m = 5       # Number of low-energy states to keep during the RSRG process
n = N       # Current Hilbert space dimension that starts with N

######################################################################################################################

# Storing results :
max_N = []           # To store the final system size after convergence, for each λ
min_eigenvalues = [] # To store the ground state energy after convergence, for each λ

iteration_counts = []                   # Iterations count for each λ
energy_histories = []                   # Energy history for each λ

# Solution computed with the mean_field approximation for all the λ values
mean_field_solutions = [mean_field(i) for i in lambda_values]


# Running the RSRG algorithm
# Loop over the λ of lambda_values

for i, L in enumerate(lambda_values) :
    
    print("##############################################")
    print(f"RSRG {i+1} for λ = {L}")             # Current iteration number and λ value
    print("---------------------------------------------")
    print(" ")  
    
    # Perform the RSRG algorithm for the current λ value and the initial Hamiltonian
    H_initial = hamiltonian(N, L) / N
    new_N, ground_state_energy, counts, history = RSRG(H_initial, N, n, m)
    
    # Append the new system size and the ground state energy to the results lists
    max_N.append(new_N)
    min_eigenvalues.append(ground_state_energy)
    
    # Append the iteration count and energy history to the corresponding lists
    iteration_counts.append(counts)
    energy_histories.append(history)
    
    print(f"RSRG {i+1} done") 
    print("##############################################") 
    print("\n\n")  

# Computing the difference between the solutions from the RSRG algorithm and the mean-field approximation solution
difference = [M - E for M, E in zip(mean_field_solutions, min_eigenvalues)]

######################################################################################################################

# Plotting the solutions from the RSRG algorithm and the Mean-field approximation :
plt.figure(figsize = (8, 6))

plt.plot(lambda_values, min_eigenvalues, label = "RSRG solutions", color = "red")
plt.plot(lambda_values, mean_field_solutions, label = "Mean-field solutions", color = "blue")

plt.xlabel(r"$\lambda$", fontsize = 16)
plt.ylabel(r"$\frac{E_0}{N}$", fontsize = 16, rotation = 0, labelpad = 15)

plt.grid()
plt.legend()

plt.savefig("solutions.pdf")
plt.show()

#---------------------------------------------------------------------------------------------------------------------

# Plotting the difference between the Mean-field solutions and the RSRG algorithm solutions 
plt.figure(figsize = (8, 6))

plt.plot(lambda_values, difference, color = "blue", label = r"$\frac{E_0}{N}_{MF} - \frac{E_0}{N}_{RSRG}$")
plt.axhline(0, color = 'black')

plt.xlabel(r'$\lambda$', fontsize = 16)
plt.ylabel(r'$\frac{E_0}{N}$', fontsize = 16, rotation = 0, labelpad = 15)

plt.grid()
plt.legend()

plt.savefig("difference.pdf")
plt.show()

#---------------------------------------------------------------------------------------------------------------------

# Plotting the number of iterations needed for convergence depending on λ
plt.figure(figsize = (8, 6))

plt.plot(lambda_values, iteration_counts, 'o', color = 'purple')

plt.xlabel(r'$\lambda$', fontsize = 16)
plt.ylabel('Number of iterations \nto reach the convergence criterion', fontsize = 16, labelpad = 15)

plt.grid()
plt.savefig("iterations_counts.pdf")

plt.show()

#---------------------------------------------------------------------------------------------------------------------

# Plotting the energy history through the iterations for specific λ

# To plot this values, set before running the code : 'lambda_steps = 7' and 'lambda_min, lambda_max = -3, 3'
selected_lambdas = [-3., -2., -1., 0., 1., 2., 3.]  

for i, L in enumerate(lambda_values) :
    if round(L, 2) in selected_lambdas :
        
        plt.figure(figsize = (8, 6))
        plt.plot(energy_histories[i], '.-', label = fr'$\lambda = {L:.1f}$', color = "blue")
        
        plt.xlabel('Number of iterations', fontsize = 16)
        plt.ylabel(r'$\frac{E_0}{N}$', fontsize = 16, rotation = 0, labelpad = 15)

        plt.xlim(-1, 18)
        plt.ylim(-0.25, 0.05)

        plt.grid()
        plt.legend()

        plt.savefig(f"energy_history_{L}.pdf")
        plt.show()

######################################################################################################################

# Not done : 
# 2) Optional : Compute the ground state energy as a function of λ using the INFINITE DMRG algorithm.