import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from NCA.NCA_utils import periodic_padding
from NCA import PDE_solver

# Assuming the PDE_solver class is already defined above

# Define the Gray-Scott equations as the function F
def gray_scott_F(X, Xdx, Xdy, Xdd, F=0.06230, k=0.06268, Du=0.1, Dv=0.05):
    u, v = X[..., 0:1], X[..., 1:2]
    uxx, vxx = Xdd[..., 0:1], Xdd[..., 1:2]
    
    du_dt = Du * uxx - u * v * v + F * (1 - u)
    dv_dt = Dv * vxx + u * v * v - (F + k) * v
    
    return tf.concat([du_dt, dv_dt], axis=-1)

# Parameters
size = [128, 128]
N_CHANNELS = 2
N_BATCHES = 1
iterations = 1024
step_size = 1.0

# Initialize the PDE solver
solver = PDE_solver(F=gray_scott_F, N_CHANNELS=N_CHANNELS, N_BATCHES=N_BATCHES, size=size, PADDING="periodic")

# Initial condition: u and v with a small random perturbation around 1 and 0, respectively
u = np.ones((N_BATCHES, size[0], size[1], 1))
v = np.zeros((N_BATCHES, size[0], size[1], 1))

# Add a small random perturbation to u and v
u += 0.01 * np.random.random(u.shape)
v += 0.01 * np.random.random(v.shape)

# Introduce a small square in the center of the grid with different concentrations
square_size = 10
u[:, size[0]//2-square_size:size[0]//2+square_size, size[1]//2-square_size:size[1]//2+square_size, 0] = 0.5
v[:, size[0]//2-square_size:size[0]//2+square_size, size[1]//2-square_size:size[1]//2+square_size, 0] = 0.25

# Set initial condition
initial_condition = np.concatenate([u, v], axis=-1)

# Run the solver
trajectory = solver.run(iterations=iterations, step_size=step_size, initial_condition=initial_condition)

# Visualization
def plot_gray_scott(u, v, step):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(u, cmap='inferno')
    ax[0].set_title(f'U at step {step}')
    ax[0].axis('off')
    ax[1].imshow(v, cmap='inferno')
    ax[1].set_title(f'V at step {step}')
    ax[1].axis('off')
    plt.show()

# Plot the initial condition
plot_gray_scott(initial_condition[0, ..., 0], initial_condition[0, ..., 1], step=0)

# Plot the states every 100 iterations
for i in range(0, iterations + 1, 100):
    plot_gray_scott(trajectory[i, 0, ..., 0], trajectory[i, 0, ..., 1], step=i)

# Plot the final state
plot_gray_scott(trajectory[-1, 0, ..., 0], trajectory[-1, 0, ..., 1], step=iterations)