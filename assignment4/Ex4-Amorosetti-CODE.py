# Assignment 4

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.special import hermite, factorial
from scipy.optimize import curve_fit
import time

#############################################################################
# Functions definitions

# We implement a function to compute the eigenvalues and eigenfunctions 
# in the context of the quantum harmonic oscillator problem 

def quantum_harmonic_oscillator(N, a, b, omega, higher_order = False) :
    
    # N : number of grid points for discretization
    # a : left boundary of x
    # b : right boundary of x
    
    # omega is the oscillator frequency
    
    # higher_order : if False, use 2nd order finite difference method
    #                if True, use higher order finite difference method
    

    # Discretization of the x-axis
    x = np.linspace(a, b, N)
    
    # Discretization space between each point
    dx = x[1] - x[0]
    
    # Potential matrix V
    V = 0.5 * omega**2 * np.diag(x**2)
    
    # Kinetic energy matrix T, using the finite difference method
    if higher_order :
        
        main_diag = (5/2) * np.ones(N) / (dx**2)
        off_diag1 = -4/3 * np.ones(N-1) / (dx**2)
        off_diag2 = 1/12 * np.ones(N-2) / (dx**2)
        T = np.diag(main_diag) + np.diag(off_diag1, k=1) + np.diag(off_diag1, k=-1)
        T += np.diag(off_diag2, k=2) + np.diag(off_diag2, k=-2)
        
    else :
        
        main_diag = 2 * np.ones(N) / (dx**2)
        off_diag = -1 * np.ones(N-1) / (dx**2)
        T = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    
    T *= 0.5  # Factor of 1/2 of the kinetic energy term
    
    # Hamiltonian H : Kinetic + Potential 
    H = T + V
    
    # Solving the quantum harmonic oscillator problem to find the eigenvalues and eigenfunctions
    eigenvalues, eigenfunctions = eigh(H)
    
    return x, eigenvalues, eigenfunctions

# We implement a function to compute the nth analytical eigenfunction
# of the same problem to compare them with the ones found with the discretization technique

#---------------------------------------------------------------------------

def analytical_eigenfunction(n, x, omega=1.0) :
    
    # n : quantum number (starting from 0), to compute the nth analytical eigenfunction
    # x : array of points for which the analytical function will be computed
    # omega : oscillator frequency  (1 by default)

    norm_factor = (1 / np.sqrt(2**n * factorial(n))) * (omega / np.pi)**0.25
    analytical_fct = norm_factor * np.exp(-omega * x**2 / 2) * hermite(n)(np.sqrt(omega) * x)
    
    return analytical_fct

#############################################################################
# Solving the problem and computing the errors

# Parameters : 

N = 3000        # Number of grid points
a, b = -5, 5    # Range of x

# Array of points for which the analytical function will be computed (only for plot comparison)
x_analytical = np.linspace(a, b, 3000)

omega = 1.0          # Oscillator frequency
higher_order = True  # Use second-order (False) or higher order (True) finite difference

number = 16 # Number of eigenvalues and eigenfuntions computed

#---------------------------------------------------------------------------

# Solving the quantum harmonic oscillator problem : 
x, eigenvalues, eigenfunctions = quantum_harmonic_oscillator(N, a, b, omega, higher_order)

# Analytical eigenvalues 
n = np.arange(len(eigenvalues))
analytical_eigenvalues = omega * (n + 0.5)

# Analytical eigenfunctions
analytical_functions = [analytical_eigenfunction(i, x_analytical, omega) for i in range(number)]

# Normalize and harmonize the computed eigenfunctions with analytical ones 
# (because these changes on the phase don't affect the physics)
adjusted_eigenfunctions = []

for i in range(number) :
    
    # Adjust phase if the initial sign doesn't match
    if np.sign(eigenfunctions[0, i]) != np.sign(analytical_functions[i][0]):
        adjusted_eigenfunction = -eigenfunctions[:, i]
    else:
        adjusted_eigenfunction = eigenfunctions[:, i]
    
    # Normalizing the amplitude
    max_computed = np.max(np.abs(adjusted_eigenfunction))
    max_analytical = np.max(np.abs(analytical_functions[i]))
    
    normalized_eigenfunction = adjusted_eigenfunction * (max_analytical / max_computed)
    
    adjusted_eigenfunctions.append(normalized_eigenfunction)

#---------------------------------------------------------------------------

# Saving the runtimes for different number of grid points, for 2nd and also higher order methods
# Saving the eigenvalues each time

grid_points_time = np.logspace(np.log10(16), np.log10(3000), 15, dtype=int)
runtime_2nd, runtime_higher = [], []
eigenvalues_2nd, eigenvalues_higher = [], []

for N in grid_points_time :
    
    # 2nd order
    start = time.time()
    _, e_2nd, _ = quantum_harmonic_oscillator(N, a, b, omega, higher_order = False)
    runtime_2nd.append(time.time() - start)
    eigenvalues_2nd.append(e_2nd[:number])

    # Higher order
    start = time.time()
    _, e_high, _ = quantum_harmonic_oscillator(N, a, b, omega, higher_order = True)
    runtime_higher.append(time.time() - start)
    eigenvalues_higher.append(e_high[:number])
    
# Performing a fit a these results
def fit_1(x, a, b) :
    # Power-law
    return a * x**b 

params_1_2nd, _ = curve_fit(fit_1, grid_points_time, runtime_2nd)
params_1_higher, _ = curve_fit(fit_1, grid_points_time, runtime_higher)

fit_1_2nd = fit_1(grid_points_time, *params_1_2nd)
fit_1_higher = fit_1(grid_points_time, *params_1_higher)
    
#---------------------------------------------------------------------------

# Computing the erros between the computed eigenvalues and the analytical ones 
# for the same number of points (3000) 

errors_eigenvalues_2nd = [np.abs(eigenvalues_2nd[-1] - analytical_eigenvalues[:number])]
errors_eigenvalues_higher = [np.abs(eigenvalues_higher[-1] - analytical_eigenvalues[:number])]

# eigenvalues_2nd[-1] and eigenvalues_higher[-1] bevause these are 2 lists storing the N lists of the 16 eigenvalues, 
# each computed for a different N (number of grid points).
# So we take the last list of these lists ([-1]) because it's the one with computed with the highest N (3000)

# Checking that errors_eigenvalues_2nd is an 1D array (for the plot later)
errors_eigenvalues_2nd = np.array(errors_eigenvalues_2nd).flatten()
errors_eigenvalues_higher = np.array(errors_eigenvalues_higher).flatten()

# Performing a fit of these results
def fit_2(x, a, b) :
    # Power-law
    return a * x**b 

# def fit_2(x, a, b) :
#     # Exponential
#     return a * np.exp(b*x) 

params_2_2nd, _ = curve_fit(fit_2, range(number)[1:], errors_eigenvalues_2nd[1:])
params_2_higher, _ = curve_fit(fit_2, range(number)[1:], errors_eigenvalues_higher[1:])

fit_2_2nd = fit_2(range(number)[1:], *params_2_2nd)
fit_2_higher = fit_2(range(number)[1:], *params_2_higher)

#---------------------------------------------------------------------------

# Computing the eigenvalue error for each quantum number, for different grid points values
# both for 2nd and higher order methods

grid_points = np.logspace(np.log10(16), np.log10(3000), 15, dtype=int)

# we start at log10(16) because the number of eigenvalues returned by the functions is equal 
# to th enumber of gridpoints, and we compare with the 16 first analytical eigenvalues,
# so we need at least 16 computed eigenvalues

errors_2nd = []
errors_higher = []

for N in grid_points :
    
    _, e_2nd, _ = quantum_harmonic_oscillator(N, a, b, omega, higher_order=False)
    _, e_high, _ = quantum_harmonic_oscillator(N, a, b, omega, higher_order=True)
    
    errors_2nd.append(np.abs(e_2nd[:number] - analytical_eigenvalues[:number]))
    errors_higher.append(np.abs(e_high[:number] - analytical_eigenvalues[:number]))

errors_2nd = np.array(errors_2nd)
errors_higher = np.array(errors_higher)
    
#############################################################################  
# Plots and prints of the results

# Comparison of eigenvalues (prints)
print("Computed and analytical eigenvalues :")
for i in range(number):
    print(f"n={i} : Computed = {eigenvalues[i]:.4f}, Analytical = {analytical_eigenvalues[i]:.4f}")
    
#---------------------------------------------------------------------------
    
# Plotting the computed and analytical eigenfunctions 
fig, axes = plt.subplots(4, 4, figsize=(16, 10))

for i, ax in enumerate(axes.flat) :  # Iterate over the subplots
    ax.plot(x, adjusted_eigenfunctions[i], label=rf"$\Psi_{{{i}}}(x)$, $E_{{{i}}}={eigenvalues[i]:.2f}$ (computed)", color='b')
    ax.plot(x_analytical, analytical_functions[i], "--", label=rf"$\phi_{{{i}}}(x)$, $E_{{{i}}}={analytical_eigenvalues[i]:.2f}$ (analytical)", color='r')

    ax.set_xlabel("x")
    ax.set_ylabel("Wavefunction")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend()
    ax.grid(True)
    ax.set_title(f"Eigenfunction {i}")

plt.tight_layout()
plt.savefig('all_eigenvalues_eigenfunctions.pdf')
plt.show()

#---------------------------------------------------------------------------

# Plotting the runtimes depending on the number of grid points 
plt.figure(figsize=(8, 6))

plt.loglog(grid_points_time, runtime_2nd, label="Second order FDM", linestyle="none", marker='o', color="blue")
plt.loglog(grid_points_time, runtime_higher, label="Higher order FDM", linestyle="none", marker='x', color="red")

# Plotting the fit
plt.plot(grid_points_time, fit_1_2nd, label=fr"Fit (Second order FDM) : ${params_1_2nd[0]:.2e} \cdot x^{{{params_1_2nd[1]:.2f}}}$", color="lightblue", linestyle='solid')
plt.plot(grid_points_time, fit_1_higher, label=fr"Fit (Higher order FDM) : ${params_1_higher[0]:.2e} \cdot x^{{{params_1_higher[1]:.2f}}}$", color="tomato", linestyle=(0, (5, 10)))

plt.xlabel("Number of grid points")
plt.ylabel("Runtime (s)")

plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.title("Runtimes comparison")

plt.savefig('runtimes.pdf')
plt.show()

#---------------------------------------------------------------------------

# Eigenvalue stability / values depending on number of grid points
plt.figure(figsize=(8, 10))

# To avoid making more plots than eigenvalues computed
max_eigen_to_plot = min(9, len(eigenvalues_2nd[0]))  

for i in range(max_eigen_to_plot):
    plt.plot(grid_points_time, [e[i] for e in eigenvalues_2nd], label=f"n={i}, Second order FDM", linestyle="none", marker='o')
    plt.plot(grid_points_time, [e[i] for e in eigenvalues_higher], label=f"n={i}, Higher order FDM", linestyle="none", marker='x')
    plt.axhline(y=analytical_eigenvalues[i], color = 'dimgrey', linestyle = '--')

plt.xscale('log')
plt.xlabel("Number of grid points")
plt.ylabel("Eigenvalue")
plt.legend(loc='upper right')
plt.grid(True, which="both", linestyle="--")
plt.title("Eigenvalue stability")
plt.savefig('error_eigenvalues_gridpoints.pdf')
plt.show()

#---------------------------------------------------------------------------

# Computed eigenvalue error (compared to analytical one) depending on the number of grid points
# Only for the first and last quantum number

# First quantum number (n = 0)
plt.figure(figsize=(8, 6))

plt.loglog(grid_points, errors_2nd[:, 0], label="Second order FDM", marker='o', linestyle="none")
plt.loglog(grid_points, errors_higher[:, 0], label="Higher order FDM", marker='x', linestyle="none")
plt.xlabel("Number of grid points")
plt.ylabel("Absolute difference")
plt.title("Error in eigenvalue (n = 0)")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.savefig('error_first_eigenvalue_gridpoints.pdf')
plt.show()

# Last quantum number (n = last)
last_n = number - 1

plt.figure(figsize=(8, 6))

plt.loglog(grid_points, errors_2nd[:, last_n], label="Second order FDM", marker='o', linestyle="none")
plt.loglog(grid_points, errors_higher[:, last_n], label="Higher order FDM", marker='x', linestyle="none")
plt.xlabel("Number of grid points")
plt.ylabel("Absolute difference")
plt.title(f"Error in eigenvalue (n = {last_n})")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.savefig('error_last_eigenvalue_gridpoints.pdf')

plt.show()

#---------------------------------------------------------------------------

# Computed eigenvalue error (compared to analytical one) depending on the quantum number
plt.figure(figsize=(8, 6))

# Plot
plt.plot(range(number), errors_eigenvalues_2nd, label="Second order FDM", color="blue", linestyle="none", marker='o')
plt.plot(range(number), errors_eigenvalues_higher, label="Higher order FDM", color="red", linestyle="none", marker='x')

# Plotting the fit (exp fit)
# plt.plot(range(number), fit_2_2nd, label=fr"Fit (Second order FDM) : ${params_2_2nd[0]:.2e} \cdot exp(x \cdot {{{params_2_2nd[1]:.2f}}})$", color="lightblue", linestyle='solid')
# plt.plot(range(number), fit_2_higher,label=fr"Fit (Higher order FDM) : ${params_2_higher[0]:.2e} \cdot exp(x \cdot{{{params_2_higher[1]:.2f}}})$", color="tomato", linestyle=(0, (5, 10)))

# Plotting the fit (power law fit)
plt.plot(range(number)[1:], fit_2_2nd, label=fr"Fit (Second order FDM) : ${params_2_2nd[0]:.2e} \cdot x^{{{params_2_2nd[1]:.2f}}}$", color="lightblue", linestyle='solid')
plt.plot(range(number)[1:], fit_2_higher,label=fr"Fit (Higher order FDM) : ${params_2_higher[0]:.2e} \cdot x^{{{params_2_higher[1]:.2f}}}$", color="tomato", linestyle=(0, (5, 10)))

#plt.yscale('log')

plt.xlabel("Quantum number (n)")
plt.ylabel("Absolute difference")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.title("Error in eigenvalues")
plt.savefig('error_eigenvalues_quantum_number.pdf')

plt.show()

#---------------------------------------------------------------------------

