# Assignment 3, Exercise 1

import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Compile list with the flags needed to compile the Fortran .f90 code for matrix multiplication
# We don't use optimization flags like the O flags because they make MATMUL() perform worse than the implemented methods

compile_command = ["gfortran", "-fopenmp", "-march=native", "-mtune=native", 
                   "-mavx512f", "-mavx512dq", "-ftree-vectorize", "-ftree-loop-vectorize",
                   "-o", "Ex3-Amorosetti-CODE-1-1.exe", "Ex3-Amorosetti-CODE-1-1.f90"]

# Executing the command
subprocess.run(compile_command, check=True) 

# Min and Max values for the matrix sizes used as input of the executable
N_min = 10
N_max = 5000

# Number of sizes between N_min and N_max used as input of the executable
num = 100

# List of num logarithmically spaced values between N_min and N_max
N_values = np.logspace(np.log10(N_min), np.log10(N_max), num, dtype=int)

# List of the different multiplication methods
methods = ["Standard", "Column-wise", "Standard, parallelized", "Column-wise, parallelized", "MATMUL()"]

# Creation of a dictionary to store execution times, for each method
execution_times = {method: [] for method in methods}

# Running the compiled executable with different matrix sizes : 

# For each size N of the N_values list
for N in N_values :
    
    # For each method, we compute the matrix multiplication with that size N
    for method in methods :
        
        print("--------------------------------------------------")
        print(method, "matrix multiplication")
        print("Matrix size :", N)
       
        # Executing the program with the current matrix size N
        result = subprocess.run(["./Ex3-Amorosetti-CODE-1-1.exe", method, str(N)], check=True, capture_output=True, text=True)
        
        # The execution time measured by the Fortran program is the last line of the executable output, that we convert to a float
        exec_time = float(result.stdout.strip().split()[-1])
        
        # Saving the execution time in the dictionnary
        execution_times[method].append(exec_time)
        
        print("Execution time :", exec_time, "seconds")
        print("--------------------------------------------------")
        
        # Saving the execution time to a text file for each method
        with open(f"{method}_execution_times.txt", "a") as file:
            file.write(f"N={N}, time={exec_time}\n")
            
            
# Definition of the function used to fit the scaling of the execution time for each method, to estimate their complexity
def polynomial_fit(x, a, b):
    return a*(x**b)

# Used to generate colors used for the plot (to use the same color for a method and its associated fit)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# Plotting the results for each method on a same figure
plt.figure(figsize=(8, 8))

# For loop iteration on each method and its associated times on the execution_times dictionnary : 
# i is a counter only used for associating a different color for each method

for i, (method, times) in enumerate(execution_times.items()):
    
    # Choosing a color for the current method in the for loop
    color = colors[i % len(colors)]
    
    # Plotting the execution times for that method
    plt.plot(N_values, times, 'o', label=f"{method} matrix multiplication", markersize=4, color = color, alpha=0.5)
    
    # Fitting a polynomial function to the execution times of that method
    params, approx_cov = curve_fit(polynomial_fit, N_values, times)
    a, b = params
    print(params)
    
    # Representation of the parameters (for a readable form for the legend plot)
    a_sci = "{:.2e}".format(a) # a in scientific notation with 2 significant digits
    b_sci = "{:.2}".format(b)  # b with 2 decimals
    
    # Plotting the fit of that method 
    plt.plot(N_values, polynomial_fit(N_values,a, b), linestyle='--', lw=1, alpha=0.75, color = color, label=rf"Fit : $({a_sci}) \cdot N^{{{b_sci}}}$")

# Plot on a log-log scale, in order to show the power law behavior of the complexity
plt.xscale("log")
plt.yscale("log")

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Matrix Size N", fontsize=16)
plt.ylabel("Execution time (s)", fontsize=16)
plt.grid(True)
plt.legend(fontsize=10)
plt.title("Scaling of Matrix Multiplication Methods", fontsize=18)

# Saving the figure as a .pdf file
plt.savefig('matrix_multiplication_times.pdf', bbox_inches='tight')

plt.show()
