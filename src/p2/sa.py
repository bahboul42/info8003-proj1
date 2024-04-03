from section4 import *
import numpy as np
import matplotlib.pyplot as plt
from section5 import OptimalAgent, QNetwork
from section2 import PolicyEstimator
import torch
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the saved model weights
model_path = 'workinnn.pth'

# Instantiate the model
model = QNetwork(input_size=3, output_size=1, hidden_sizes=[8, 16, 32, 16, 8])

# Load the model weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
# Set the model to evaluation mode
model.eval()
domain = Domain() # Create the environment
domain.sample_initial_state() # Sample an initial state

agent = OptimalAgent(model) # Create the agent
policy_est = PolicyEstimator(domain, agent)

n_initials = 50 # Number of initial states
N = 300 # Horizon

all_returns = policy_est.policy_return(N, n_initials) # Get all the estimated expected returns

policy_est.plot_return(all_returns, path='.') # Plot the returns for the 50 initial states

# Inspecting the final expected return of all trajectories to know in more detail
count_pos = 0 # Number of times the expected return is positive
count_neg = 0 # Number of times the expected return is negative
for i in range(n_initials):
    if all_returns[i, -1] > 0:
        count_pos += 1
    elif all_returns[i, -1] < 0:
        count_neg += 1
print(f'Number of successful trajectories: {count_pos}')
print(f'Number of unsuccessful trajectories: {count_neg}')







# iters = 1000
# iters2 = 100000

# X, y, z = get_set(mode='randn', n_iter=int(iters2))
# X2, y2, z2 = get_set(mode='episodic', n_iter=int(iters))

# # Plotting the two sets of points to compare their distributions
# plt.figure(figsize=(14, 6))

# # Plot for 'randn' mode
# plt.subplot(1, 2, 1)
# scatter1 = plt.scatter(X[:, 0], X[:, 1], alpha=0.3, c=y, cmap='viridis')
# plt.colorbar(scatter1, label='Reward (-1, 0, 1)')
# plt.title('Random Mode (randn)')
# plt.xlabel('X')
# plt.ylabel('Y')

# # Plot for 'episodic' mode
# plt.subplot(1, 2, 2)
# scatter2 = plt.scatter(X2[:, 0], X2[:, 1], alpha=0.3, c=y2, cmap='viridis')
# plt.colorbar(scatter2, label='Reward (-1, 0, 1)')
# plt.title('Episodic Mode')
# plt.xlabel('X')
# plt.ylabel('Y')

# plt.tight_layout()
# plt.show()
