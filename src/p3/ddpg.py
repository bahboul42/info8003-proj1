import gymnasium as gym
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn

# Setting the random seed to obtain reproducible results
seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value) if torch.cuda.is_available() else None
np.random.seed(seed_value)

# Setting the parameter values
lr_actor = 0.001
lr_critic = 0.001
tau = 0.01
n_transitions = 1000
n_episodes = 10000
noise_std = 0.5
buffer_size = 50000
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, double=False):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.double = double

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_actor)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))
        if not self.double:
            x = 3 * x # Action space is [-3, 3] for simple pendulum
        return x

    def update(self, actor, tau):
        for target_param, param in zip(self.parameters(), actor.parameters()):
            target_param.data = (tau * param.data + (1.0-tau) * target_param.data)

    def train(self, critic, states):
        self.optimizer.zero_grad()

        # Temporarily set requires_grad to False for critic parameters
        for param in critic.parameters():
            param.requires_grad = False

        critic_output = critic(states, self.forward(states))
        loss = -critic_output.mean()
        loss.backward()

        # Restore requires_grad to True for critic parameters
        for param in critic.parameters():
            param.requires_grad = True

        self.optimizer.step()

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, double=False):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

        self.relu = nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_critic)

        self.loss_fct = nn.MSELoss()

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        if not self.double:
            x = self.relu(x) # No negative rewards for simple pendulum
        return x

    def update(self, critic, tau):
        for target_param, param in zip(self.parameters(), critic.parameters()):
            target_param.data = (tau * param.data + (1.0-tau) * target_param.data)

    def train(self, states, actions, y):
        self.optimizer.zero_grad()
        y_pred = self.forward(states, actions)
        loss = self.loss_fct(y_pred, y)
        loss.backward()
        self.optimizer.step()

class ReplayBuffer():
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.idx = 0

    def size(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.idx] = (state, action, reward, next_state, done)
        self.idx = (self.idx + 1) % self.max_size

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in idxs:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state.unsqueeze(0))
            actions.append(action.unsqueeze(0))
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return torch.cat(states, dim=0), torch.cat(actions, dim=0), torch.tensor(rewards, dtype=torch.float32).to(device), torch.tensor(next_states, dtype=torch.float32).to(device), torch.tensor(dones, dtype=torch.float32).to(device)

def ddpg(env, n_episodes, n_transitions, noise_std, buffer_size, batch_size, tau, double=False):
    buffer = ReplayBuffer(buffer_size)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim, double).to(device)
    critic = Critic(state_dim, action_dim, double).to(device)

    target_actor = deepcopy(actor).to(device)
    target_critic = deepcopy(critic).to(device)

    hasConverged = False

    for episode in range(n_episodes):
        if hasConverged:
            break

        state, _ = env.reset()
        noise_std = noise_std * (0.1 ** (1 / n_episodes))

        # total_rew = 0

        for _ in range(n_transitions):
            state = torch.tensor(state, dtype=torch.float32).to(device)

            action = actor(state).detach() + noise_std * torch.randn(action_dim).to(device)
            action = torch.clamp(action, -3, 3)

            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = 1 if terminated else 0

            # total_rew += reward

            buffer.add(state, action, reward, next_state, done)

            if buffer.size() > batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                rewards = rewards.unsqueeze(dim=-1)
                dones = dones.unsqueeze(dim=-1)

                y = rewards + target_critic(next_states.to(device), target_actor(next_states.to(device))) * (1 - dones)

                critic.train(states.to(device), actions.to(device), y.to(device))

                actor.train(critic, states.to(device))

                target_actor.update(actor, tau)
                target_critic.update(critic, tau)

            if terminated or truncated:
                break

            state = next_state

        if (episode + 1) % 100 == 0:
            torch.save(actor.state_dict(), "actor.pth")
            torch.save(critic.state_dict(), "critic.pth")

            print(f"Episode {episode + 1}/{n_episodes}")

            state, _ = env.reset()
            total_rew = 0
            for _ in range(n_transitions):
                state = torch.tensor(state, dtype=torch.float32).to(device)
                action = actor(state).detach()
                next_state, reward, terminated, _, _ = env.step(action.cpu().numpy())
                total_rew += reward
                if terminated:
                    break
                state = next_state

            print(f"Total reward: {total_rew}")
            if total_rew == 1000:
                hasConverged = True
                print('converged')

    return actor, critic

def main():
    env = gym.make("InvertedPendulum-v4")
    actor, critic = ddpg(env, n_episodes, n_transitions, noise_std, buffer_size, batch_size, tau)
    env.close()

if __name__ == "__main__":
    main()

