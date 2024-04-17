import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from itertools import product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from tqdm import tqdm

import gymnasium as gym

# CONSTANTS
lr = 1e-3
BATCH_SIZE = 256
NUM_ACTIONS = 100
NUM_ITERS = 200
ACTIONS = torch.linspace(-3, 3, NUM_ACTIONS).to(device)

# 1. 1-step transitions
# 2. Create an input/output set with 1 step transitions and fitted q from last horizon
# 3. Train a model on the input/output set
class QNetwork(nn.Module):
    ''' Neural network for Q-learning '''
    def __init__(self, input_size, output_size=1, hidden_sizes=[16, 32, 64, 128, 128, 64, 32, 16]):
        super(QNetwork, self).__init__()
        layers = []
        input_dim = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
def nn_fitted_q_iteration(env, data, n_q=NUM_ITERS, batch_size=BATCH_SIZE):
    ''' Performs Fitted Q Iteration on the given domain and transitions with a neural network. '''
    num_features = env.observation_space.shape[0]
    
    X = torch.tensor(data[:, :(num_features + 1)], dtype=torch.float32).to(device)
    y = torch.tensor(data[:, (num_features + 1)], dtype=torch.float32).view(-1, 1).to(device)
    z = torch.tensor(data[:, (num_features + 3):], dtype=torch.float32).to(device)
    z = set_z(z)
    z = z.view(X.shape[0], (num_features + 1) * NUM_ACTIONS)

    # Creating a dataset and dataloader for mini-batch training
    dataset = TensorDataset(X, y, z)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = QNetwork(input_size=X.shape[1], output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10,)

    model.train()

    for X_batch, y_batch, z_batch in dataloader:

        optimizer.zero_grad()
        l_pred = model(X_batch)
        loss = criterion(l_pred, y_batch)
        loss.backward()
        optimizer.step()

    for iter in tqdm(range(n_q-1)):

        with torch.no_grad():
            l_pred = model(X)

        for X_batch, y_batch, z_batch in dataloader:
            
            optimizer.zero_grad()

            v_z = z_batch.view(z_batch.shape[0] * NUM_ACTIONS, num_features + 1)
            v_z = model(v_z)
            v_z = v_z.view(-1, NUM_ACTIONS)
            concat = torch.max(v_z)

            out = y_batch + concat

            l_pred_batch = model(X_batch)

            loss = criterion(l_pred_batch, out)
            loss.backward()
            optimizer.step()

        scheduler.step(loss)
        with torch.no_grad():
            r_pred = model(X)
            e = criterion(l_pred, r_pred) # APPEND TO TENSOR ?

    # save mode
        if iter % 20 == 0:
            print(f"Iteration {iter} Loss: {e.item()}\r")
            torch.save(model.state_dict(), f'models/fqi_{iter}.pth')
    return model

def set_z(states, actions=ACTIONS):
    """
        Prepare state-action pairs for all actions and all states in the batch.

        Parameters:
            states (torch.Tensor): The state tensor of shape [batch_size, state_dim].
            actions (torch.Tensor): A linspace tensor of actions of shape [num_actions].

        Returns:
            torch.Tensor: A tensor of shape [batch_size * num_actions, state_dim + 1] where each state
                        is repeated for each action and concatenated with the action.
    """
    # Number of actions and batch size
    num_actions = actions.size(0)
    batch_size = states.size(0)
    
    # Repeat each state for each action
    states = states.unsqueeze(1).repeat(1, num_actions, 1)
    states = states.view(batch_size * num_actions, -1)

    # Repeat actions for the whole batch
    actions = actions.repeat(batch_size, 1).view(batch_size * num_actions, 1)
    
    # Concatenate states with actions
    state_action_pairs = torch.cat((states, actions), dim=1)
    
    return state_action_pairs


class OptimalAgent:
    ''' Agent that selects the optimal action based on a given model. '''
    def __init__(self, model):
        self.model = model

    def get_action(self, state):
        ''' Returns the optimal action for a given state. '''
        with torch.no_grad():

            state = torch.tensor(state, dtype=torch.float32).to(device)
            state = state.unsqueeze(0)
            state = set_z(state)

            q_values = self.model(state)
            max_q_i = torch.argmax(q_values)

            action = ACTIONS[max_q_i]
        return action.item()
            
def generate_transitions(env, num_transitions, X=[], y=[], z=[]):
    observation, info = env.reset(seed=42)

    # Define the shape of the array you need based on your environment's observation space and action space
    num_features = env.observation_space.shape[0]
    action_dim = 1  # Assuming a single continuous action for simplicity
    data = np.zeros((num_transitions, num_features + action_dim + 1 + 1 + num_features))  # state + action + reward + done + next_state

    for i in range(num_transitions):
        action = env.action_space.sample()
    
        next_observation, reward, terminated, truncated, info = env.step(action)
            
        done = 1 if terminated else 0
        
        data[i] = np.hstack((observation, action, reward, done, next_observation))

        if terminated or truncated:
            observation, info = env.reset()
        else:
            observation = next_observation

    return data


def main():
    # Initialize the environment
    render_mode = None
    # render_mode = 'human'
    env = gym.make("InvertedPendulum-v4", render_mode=render_mode)  # 'human' mode is for visual, use None for faster computation

    # Number of transitions generated
    num_transitions = 700000

    # Generate the set of transitions
    transitions = generate_transitions(env, num_transitions)

    model = nn_fitted_q_iteration(env, transitions)
    agent = OptimalAgent(model)

    env.close()
    env = gym.make("InvertedPendulum-v4", render_mode='human')  # 'human' mode is for visual, use None for faster computation

    observation, info = env.reset(seed=42)
    done = False
    total_reward = 0

    while not done or truncated:
        action = agent.get_action(observation)
        print(action)

        next_observation, reward, terminated, truncated, info = env.step([action])
            
    env.close()    

if __name__ == "__main__":
  main()