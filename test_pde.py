import numpy as np
import tensorflow as tf
from tqdm import tqdm
from NCA.NCA_utils import periodic_padding
import matplotlib.pyplot as plt
import os
from NCA import PDE_solver

# Define the PDE function F for the heat equation
def heat_equation(X, Xdx, Xdy, Xdd):
    alpha = 0.01  # Thermal diffusivity
    return alpha * Xdd  # Laplacian term only

# Initialize the PDE solver
N_CHANNELS = 1
N_BATCHES = 1
size = [128, 128]
PADDING = "periodic"
F = heat_equation

solver = PDE_solver(F, N_CHANNELS, N_BATCHES, size, PADDING)

# Set initial condition (e.g., a hot spot in the center)
initial_condition = np.zeros((N_BATCHES, size[0], size[1], N_CHANNELS))
initial_condition[0, size[0]//2, size[1]//2, 0] = 1.0

# Run the solver
iterations = 1000
step_size = 0.1
trajectory = solver.run(iterations, step_size, initial_condition)

# Create a directory to save the images
os.makedirs('visualizations', exist_ok=True)

# Save the visualization results as image files
for i in range(0, iterations + 1, 100):  # Save every 100th step
    plt.imshow(trajectory[i, 0, :, :, 0], cmap='hot', interpolation='nearest')
    plt.title(f"Iteration {i}")
    plt.colorbar()
    plt.savefig(f'visualizations/iteration_{i}.png')
    plt.close()  # Close the figure to avoid memory issues
