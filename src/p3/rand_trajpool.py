import gymnasium as gym
import numpy as np

#HYPERPARAMS
render_mode = None
# render_mode = 'human'

#

env = gym.make("InvertedPendulum-v4", render_mode=render_mode)  # 'human' mode is for visual, use None for faster computation
observation, info = env.reset(seed=42)

# Define the shape of the array you need based on your environment's observation space and action space
num_steps = 1000
num_features = env.observation_space.shape[0]
action_dim = 1  # Assuming a single continuous action for simplicity
data = np.zeros((num_steps, num_features + action_dim + 1 + 1 + num_features))  # state + action + reward + done + next_state

for i in range(num_steps):

    action = np.random.uniform(-1, 1, size=(action_dim,)) # POLICY HERE
    
    next_observation, reward, terminated, truncated, info = env.step(action)
        
    done = 1 if terminated else 0
    
    data[i] = np.hstack((observation, action, reward, done, next_observation))

    if terminated or truncated:
        observation, info = env.reset()
    else:
        observation = next_observation

env.close()

X = data[:, :(num_features + 1)]
y = data[:, (num_features + 1)]
z = data[:, (num_features + 3):]

print(X.shape, y.shape, z.shape)
