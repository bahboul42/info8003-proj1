
import random

import numpy as np

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CONSTANTS
lr = 1e-3
NUM_EPISODES = int(5e3)
EPS = 1e-6
GAMMA = 1

class GaussianFeedForward(nn.Module):
    ''' Neural network for Q-learning '''
    def __init__(self, input_size=4, output_size=2, hidden_sizes=[32, 16], do_dropout=True):
        super(GaussianFeedForward, self).__init__()
        layers = []
        input_dim = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.Tanh())
            if do_dropout:
                layers.append(nn.Dropout(.2))
            input_dim = hidden_size
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x) # return mu, logvar

def reinforce(model, env, optimizer, seed, isDouble=False):
    ''' Perform REINFORCE algorithm '''
    model.train()
    for episode in tqdm(range(NUM_EPISODES)):

        observation, info = env.reset()
        done = False

        log_probs = torch.empty((0,), dtype=torch.float32)
        rewards = torch.empty((0,), dtype=torch.float32)

        while not done:

            observation = torch.tensor(observation, dtype=torch.float32).to(device)

            out = model(observation)
            mu, logvar = out[0], out[1]

            action, log_prob = sample_action(mu, logvar)
            
            action = torch.clamp(action, -1, 1) if isDouble else torch.clamp(action, -3, 3)

            next_observation, reward, terminated, truncated, info = env.step([action])
            
            done = terminated or truncated

            log_probs = torch.cat((log_prob.unsqueeze(0), log_probs), dim=0)
            rewards = torch.cat((torch.tensor(reward, dtype=torch.float32).unsqueeze(0), rewards), dim=0)
            observation = next_observation

        if episode % 100 == 0: 
            print(f"Episode: {episode}, Reward: {rewards.sum()}")
            if isDouble:
                torch.save(model.state_dict(), f"./models/dreinforce_{episode}-{seed}.pth")
            else:
                torch.save(model.state_dict(), f"./models/reinforce_{episode}-{seed}.pth")

        update(rewards, log_probs, optimizer)
    return model

def sample_action(mu, logvar):
    std = torch.exp(logvar / 2)
    distrib = Normal(mu, std)
    action = distrib.sample()
    log_prob = distrib.log_prob(action)
    return action.cpu(), log_prob.cpu()

def update(rewards, log_probs, optimizer):
    g = 0
    gs = []

    # Discounted return (backwards)
    for R in rewards:
        g = R + GAMMA * g
        gs.append(g)

    deltas = torch.tensor(gs, dtype=torch.float32)

    loss = 0
    # minimize -1 * prob * reward obtained
    for log_prob, delta in zip(log_probs, deltas):
        loss += log_prob.mean() * delta * (-1)

    # Update the policy network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def inverted_main():
    env = gym.make("InvertedPendulum-v4", render_mode=None)
    num_features = env.observation_space.shape[0]
    model = GaussianFeedForward(input_size=num_features, output_size=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.load_state_dict(torch.load("./models/reinforce_4700-42.pth"))
    seeds = [42, 43, 44, 45, 46]
    for s in seeds:
        random.seed(s)
        torch.manual_seed(s)
        np.random.seed(s)
        model = reinforce(model, env, optimizer, s)
    torch.save(model.state_dict(), "./models/reinforce.pth")
    
    model.load_state_dict(torch.load("./models/reinforce.pth"))
    env = gym.make("InvertedPendulum-v4", render_mode='human')
    observation, info = env.reset()

    model.eval()

    done = False
    rewards = 0
    while not done:

        out = model(torch.tensor(observation, dtype=torch.float32).to(device))
        mu, logvar = out[0], out[1]

        action, _ = sample_action(mu, logvar)
        next_observation, reward, terminated, truncated, info = env.step([action])
        
        rewards += reward

        done = terminated or truncated
        observation = next_observation
    print(f"Total Reward: {rewards}")


def double_inverted_main():
    env = gym.make("InvertedDoublePendulum-v4", render_mode=None)

    num_features = env.observation_space.shape[0]
    model = GaussianFeedForward(input_size=num_features, output_size=2, do_dropout=False, hidden_sizes=[16, 32, 64, 32, 16]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    seeds = [42, 43, 44, 45, 46]
    for s in seeds:
        random.seed(s)
        torch.manual_seed(s)
        np.random.seed(s)
        model = reinforce(model, env, optimizer, s, True)
    torch.save(model.state_dict(), "./models/dreinforce.pth")

if __name__ == "__main__":
    double_inverted_main()