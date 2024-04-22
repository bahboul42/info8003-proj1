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
BATCH_SIZE = 2048
NUM_ACTIONS = int(5)
NUM_ITERS = 100
N_TRANS = 1000000
ACTIONS = torch.linspace(-3, 3, NUM_ACTIONS).to(device)
GAMMA = .99

# 1. 1-step transitions
# 2. Create an input/output set with 1 step transitions and fitted q from last horizon
# 3. Train a model on the input/output set
class QNetwork(nn.Module):
    ''' Neural network for Q-learning '''
    def __init__(self, input_size, output_size=1, hidden_sizes=[32, 16]):
        super(QNetwork, self).__init__()
        layers = []
        input_dim = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.Tanh()) # Tanh ?\
            input_dim = hidden_size
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
class FQIAgent:
    def __init__(self, model):
        self.model = model

    def get_action(self, state, actions=ACTIONS):
        ''' Returns the optimal action for a given state. '''
        with torch.no_grad():

            state = torch.tensor(state, dtype=torch.float32, device=device)
            state = state.unsqueeze(0)
            state = set_z(state)

            q_values = self.model(state)
            max_q_i = torch.argmax(q_values)

            action = ACTIONS[max_q_i]
        return action.item()

def nn_fitted_q_iteration(env, data, model, n_q=NUM_ITERS, batch_size=BATCH_SIZE, actions=ACTIONS, model_name="fqi"):
    ''' Performs Fitted Q Iteration on the given domain and transitions with a neural network. '''
    num_features = env.observation_space.shape[0]
    
    X = torch.tensor(data[:, :(num_features + 1)], dtype=torch.float32).to(device)
    y = torch.tensor(data[:, (num_features + 1)], dtype=torch.float32).view(-1, 1).to(device)
    z = torch.tensor(data[:, (num_features + 3):], dtype=torch.float32).to(device)
    z = set_z(z, actions=actions)
    z = z.view(X.shape[0], (num_features + 1) * NUM_ACTIONS)
    # Creating a dataset and dataloader for mini-batch training
    dataset = TensorDataset(X, y, z)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    agent = FQIAgent(model)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
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
            concat, _ = torch.max(v_z, dim=1, keepdim=True)
            out = y_batch + concat * GAMMA

            l_pred_batch = model(X_batch)

            loss = criterion(l_pred_batch, out)
            loss.backward()
            optimizer.step()

        scheduler.step(loss)
        with torch.no_grad():
            r_pred = model(X)
            e = criterion(l_pred, r_pred) # APPEND TO TENSOR ?

    # save mode
        if iter % 10 == 0:
            print(f"Iteration {iter} Loss: {e.item()}\r")
            torch.save(model.state_dict(), f'models/{model_name}_{iter}.pth')
            envi = gym.make("InvertedDoublePendulum-v4", render_mode='human')
            done = False
            observation, info = envi.reset()
            rewards = 0
            while not done:
                action = agent.get_action(observation, actions=ACTIONS)
                next_observation, reward, terminated, truncated, info = envi.step([action])
                rewards += reward
                done = terminated or truncated
                observation = next_observation
            print(f"Total Reward earned with FQI: {rewards}")
            envi.close()

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
            
def generate_transitions(env, num_transitions, X=[], y=[], z=[]):
    observation, info = env.reset(seed=42)

    # Define the shape of the array you need based on your environment's observation space and action space
    num_features = env.observation_space.shape[0]
    action_dim = 1  # Assuming a single continuous action for simplicity
    data = np.zeros((num_transitions, num_features + action_dim + 1 + 1 + num_features))  # state + action + reward + done + next_state

    for i in range(num_transitions):
        action = env.action_space.sample()
    
        next_observation, reward, terminated, truncated, info = env.step(action)
            
        done = 1 if terminated  or truncated else 0
        
        data[i] = np.hstack((observation, action, reward, done, next_observation))

        if terminated or truncated:
            observation, info = env.reset()
        else:
            observation = next_observation

    return data

if __name__ == "__main__":
    render_mode = None
    # env = gym.make("InvertedPendulum-v4", render_mode=render_mode)  # 'human' mode is for visual, use None for faster computation
    # num_features = env.observation_space.shape[0]
    # num_transitions = N_TRANS
    # model = QNetwork(input_size=num_features+1, output_size=1).to(device)
    # transitions = generate_transitions(env, num_transitions)
    # model = nn_fitted_q_iteration(env, transitions, model)
    # env.close()

    ACTIONS = torch.linspace(-1, 1, NUM_ACTIONS).to(device)

    env = gym.make("InvertedDoublePendulum-v4", render_mode=render_mode)
    num_features = env.observation_space.shape[0]
    num_transitions = N_TRANS
    model = QNetwork(input_size=num_features+1, output_size=1, hidden_sizes=[64, 256, 64]).to(device)
    transitions = generate_transitions(env, num_transitions)
    model = nn_fitted_q_iteration(env, transitions, model, actions=ACTIONS, model_name="dfqi")
    env.close()