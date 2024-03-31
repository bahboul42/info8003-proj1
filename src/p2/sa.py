from section4 import *
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

iters = 1000
iters2 = 100000

X, y, z = get_set(mode='randn', n_iter=int(iters2))
X2, y2, z2 = get_set(mode='episodic', n_iter=int(iters))

# Plotting the two sets of points to compare their distributions
plt.figure(figsize=(14, 6))

# Plot for 'randn' mode
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X[:, 0], X[:, 1], alpha=0.3, c=y, cmap='viridis')
plt.colorbar(scatter1, label='Reward (-1, 0, 1)')
plt.title('Random Mode (randn)')
plt.xlabel('X')
plt.ylabel('Y')

# Plot for 'episodic' mode
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(X2[:, 0], X2[:, 1], alpha=0.3, c=y2, cmap='viridis')
plt.colorbar(scatter2, label='Reward (-1, 0, 1)')
plt.title('Episodic Mode')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()
