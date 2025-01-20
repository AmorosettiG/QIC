# Assignment 5

import numpy as np
import matplotlib.pyplot as plt

################################################################################
# Problem parameters

#-------------------------------------------------------------------------------
# Setting the physical constants (using reduced constants and natural units)

hbar = 1.0  # reduced Planck constant
m = 1.0     # mass of the particle
omega = 1.0 # angular frequency

#-------------------------------------------------------------------------------
# Parameters for the numerical simulation

x_min, x_max = -4, 4  # space interval
Nslice = 1000         # number of slices in space
Ntslice = 1000        # number of slices in time

T = 1000              # timescale for q0(t) 
t_sim = T             # duration of the simulation (because 0 <= t_sim <= T)

dx = (x_max - x_min) / Nslice                            # space discretization
dt = t_sim / Ntslice                                     # time discretization
x = np.linspace(x_min, x_max, Nslice, endpoint = False)  # spatial grid

# Kinetic part in the momentum space
p = (2 * np.pi / (x_max - x_min)) * np.fft.fftfreq(Nslice, d=dx)     # momentum grid discretization (grid of frequencies)

################################################################################
# Functions definitions

# Potential q0(t)
def q0(t) :
    return t / T

# Potential energy operator
def V(x, t) :
    return 0.5 * m * omega**2 * (x - q0(t))**2

# Ground state of the harmonic oscillator (n=0)
def psi0(x) :
    return (m * omega / (np.pi * hbar))**0.25 * np.exp(-m * omega * x**2 / (2 * hbar))             
#-------------------------------------------------------------------------------

# Split-step method :

def split_step(psi, x, p, dt, Ntslice) :
    
    # Array to store the evolution of the wavefunction for each t
    evolution = np.zeros((Nslice, Ntslice + 1), dtype=complex)
    evolution[:, 0] = psi # Initialize the first column with the values of Psi at the inital time


    # Loop over the time steps :
    # (normalization is printed at every step)
    
    for n in range(Ntslice) :
            
        t = n * dt  # Current time step

        # Applying the potential operator (1st half step) :
        
        Vx = V(x, t)
        psi = np.exp(-1j * Vx * dt / (2 * hbar)) * psi
        print(f"Step {n + 1} - After applying the potential operator (first half step) : Norm = {np.linalg.norm(psi) * np.sqrt(dx)}")

        # Fourier transform to the momentum space :
        
        psi_p = np.fft.fft(psi)
        print(f"Step {n + 1} - After Fourier transform : Norm = {np.linalg.norm(psi_p) * np.sqrt(dx)}")

        # Multiplying by the kinetic operator :
        
        T_p = p**2 / (2 * m)
        psi_p = np.exp(-1j * T_p * dt / hbar) * psi_p
        print(f"Step {n + 1} - After multiplying by the kinetic operator : Norm = {np.linalg.norm(psi_p) * np.sqrt(dx)}")

        # Reverse Fourier transform, back to the coordinate space :
        
        psi = np.fft.ifft(psi_p)
        print(f"Step {n + 1} - After the inverse Fourier transform : Norm = {np.linalg.norm(psi) * np.sqrt(dx)}")

        # Applying the potential operator (2nd half step) :
        
        Vx = V(x, t + dt)
        psi = np.exp(-1j * Vx * dt / (2 * hbar)) * psi
        print(f"Step {n + 1} - After applying the potential operator (second half step) : Norm = {np.linalg.norm(psi) * np.sqrt(dx)}")

        # Storing the evolved wavefunction :
        
        evolution[:, n + 1] = psi

    return evolution

################################################################################
# Solving the problem

# Initializing the wavefunction
psi = psi0(x)

# Normalizing the wavefunction
psi /= np.linalg.norm(psi) * np.sqrt(dx)  

# Computing the time evolution
evolution = split_step(psi, x, p, dt, Ntslice)

#-------------------------------------------------------------------------------
# Computing data for the plots

# Computing average position
avg_position = np.real(np.sum(x[:, None] * np.abs(evolution)**2, axis=0) * dx)

# Time array
time = np.linspace(0, t_sim, Ntslice + 1)

################################################################################
# Plots of the results

# Probability density plots, for different times

fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
time_indices = [0, Ntslice // 4, Ntslice // 2, Ntslice]

for ax, t_index in zip(axs.ravel(), time_indices) :
    
    Vx = V(x, time[t_index])
    
    ax.plot(x, np.abs(evolution[:, t_index])**2, label=f"|ψ(x,t)|² at t={time[t_index]:.2f}, T={T}", color = 'b')
    ax.plot(x, Vx, label="Potential V(x)", linestyle="--", color = 'r')
    
    ax.set_title(f"t = {time[t_index]:.2f}")
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Probability density")
    ax.set_ylim(-0.2,3)
    ax.set_xlim(-3,3)
    
    ax.legend()
    ax.grid()
    
plt.suptitle("Probability density at different times")
plt.tight_layout()
plt.savefig(f'proba_T_{T}.pdf')
plt.show()

#-------------------------------------------------------------------------------

# Average position plot

plt.figure(figsize=(10, 6))

plt.plot(time, avg_position, label=f"⟨x⟩ (T={T})", color = 'b')

plt.xlabel("Time (t)")
plt.ylabel("Average position ⟨x⟩")
plt.title("Average position depeding on the time")
plt.legend()
plt.grid()

plt.savefig(f'average_T_{T}.pdf')
plt.show()

