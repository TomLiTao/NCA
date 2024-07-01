import numpy as np
import tensorflow as tf
from tqdm import tqdm
from NCA.NCA_utils import periodic_padding
import matplotlib.pyplot as plt
from NCA import PDE_solver

# Define the PDE function F for the Gray-Scott reaction-diffusion system
@tf.function
def gray_scott_F(X, Xdx, Xdy, Xdd):
    A = X[..., 0]
    B = X[..., 1]
    
    DA = 0.1
    DB = 0.05
    alpha = 0.06230
    gamma = 0.06268

    lap_A = Xdd[..., 0]
    lap_B = Xdd[..., 1]

    dA = DA * lap_A - A * B**2 + alpha * (1 - A)
    dB = DB * lap_B + A * B**2 - (gamma + alpha) * B

    dX = tf.stack([dA, dB], axis=-1)
    return dX 

# Initialize the PDE solver
N_CHANNELS = 2
N_BATCHES = 1
size = [64, 64]
PADDING = "periodic"
F = gray_scott_F

solver = PDE_solver(F, N_CHANNELS, N_BATCHES, size, PADDING)

# Set initial condition
initial_condition = np.zeros((N_BATCHES, size[0], size[1], N_CHANNELS))
initial_condition[0, size[0]//2-5:size[0]//2+5, size[1]//2-5:size[1]//2+5, 0] = 0.5  # A
initial_condition[0, size[0]//2-5:size[0]//2+5, size[1]//2-5:size[1]//2+5, 1] = 0.25  # B

# Run the solver
iterations = 1024
step_size = 1.0
trajectory = solver.run(iterations, step_size, initial_condition)


# Save the visualization results as image files
for i in range(0, iterations + 1, 100):  # Save every 100th step
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    im0 = ax[0].imshow(trajectory[i, 0, :, :, 0], cmap='viridis', interpolation='nearest')
    ax[0].set_title(f"Iteration {i} - U")
    fig.colorbar(im0, ax=ax[0])
    
    im1 = ax[1].imshow(trajectory[i, 0, :, :, 1], cmap='viridis', interpolation='nearest')
    ax[1].set_title(f"Iteration {i} - V")
    fig.colorbar(im1, ax=ax[1])

    plt.savefig(f'Gray_Scott/Gray_Scott_simulation{i}.png')
    plt.close()  # Close the figure to avoid memory issues