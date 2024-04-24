import gymnasium as gym
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Setting the random seed to obtain reproducible results
seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value) if torch.cuda.is_available() else None
np.random.seed(seed_value)

# Setting the parameter values
lr_actor = 0.0001
lr_critic = 0.001
tau = 0.001
n_transitions = 1000
n_episodes = 50
buffer_size = 10000
batch_size = 64
gamma = 0.99
every_n = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, double=False):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)

        torch.nn.init.uniform_(self.fc3.weight, -0.003, 0.003)
        torch.nn.init.uniform_(self.fc3.bias, -0.003, 0.003)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.double = double

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_actor)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
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
        self.fc1 = nn.Linear(state_dim, 400-action_dim)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        torch.nn.init.uniform_(self.fc3.weight, -0.003, 0.003)
        torch.nn.init.uniform_(self.fc3.bias, -0.003, 0.003)

        self.relu = nn.ReLU()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_critic, weight_decay = 0.01)

        self.loss_fct = nn.MSELoss()

    def forward(self, state, action):
        x = self.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
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

def ddpg(env, n_episodes, n_transitions, noise_std, buffer_size, batch_size, tau, gamma, double=False, n_iterations=3):
    
    all_mean_rewards = []
    for i in range(1, n_iterations+1):
        print(f"Iteration {i}...")

        buffer = ReplayBuffer(buffer_size)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        actor = Actor(state_dim, action_dim, double).to(device)
        critic = Critic(state_dim, action_dim, double).to(device)

        target_actor = deepcopy(actor).to(device)
        target_critic = deepcopy(critic).to(device)

        # We don't need to compute their gradients
        for param in target_actor.parameters():
                param.requires_grad = True
        for param in target_critic.parameters():
                param.requires_grad = True

        mean_rewards = []

        for episode in range(n_episodes):
            state, _ = env.reset()
            noise_std = noise_std * (0.5 ** (1/n_episodes))

            for transition in range(1, n_transitions+1):
                state = torch.tensor(state, dtype=torch.float32).to(device)

                action = actor(state).detach()
                action = action + noise_std * torch.randn(action_dim).to(device)
                action = torch.clamp(action, -3, 3)

                next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                done = 1 if terminated else 0

                buffer.add(state, action, reward, next_state, done)

                if buffer.size() > batch_size:
                    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                    rewards = rewards.unsqueeze(dim=-1)
                    dones = dones.unsqueeze(dim=-1)

                    y = rewards + gamma * target_critic(next_states.to(device), target_actor(next_states.to(device))) * (1 - dones)

                    critic.train(states.to(device), actions.to(device), y.to(device))

                    actor.train(critic, states.to(device))

                    target_actor.update(actor, tau)
                    target_critic.update(critic, tau)

                if terminated or truncated:
                    state, _ = env.reset()
                else:
                    state = next_state

                if transition % every_n == 0:
                    if double:
                        torch.save(actor.state_dict(), f"dactor.pth")
                        torch.save(critic.state_dict(), "dcritic.pth")
                    else:
                        torch.save(actor.state_dict(), "actor.pth")
                        torch.save(critic.state_dict(), "critic.pth")

                    print(f"Transition {2*episode*every_n+transition}/{n_episodes*n_transitions}")

                    avg_rew = 0
                    avg_steps = 0
                    for _ in range(100):
                        state, _ = env.reset()
                        for timestep in range(1, n_transitions+1):
                            with torch.no_grad():
                                state = torch.tensor(state, dtype=torch.float32).to(device)
                                action = actor(state).detach()
                                next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
                                avg_rew += reward/100
                                if terminated or truncated:
                                    avg_steps += timestep/100
                                    break
                                state = next_state
                    print(f"Average reward: {avg_rew}")
                    print(f"Average number of time steps: {avg_steps}")
                    mean_rewards.append(avg_rew)

        all_mean_rewards.append(mean_rewards)
    
    return actor, critic, all_mean_rewards

def plot_rewards(all_mean_rewards, double=False):
    all_mean_rewards = np.array(all_mean_rewards)
    mean = np.squeeze(np.mean(all_mean_rewards, axis=0))
    std = np.squeeze(np.std(all_mean_rewards, axis=0))

    max_steps = n_episodes * n_transitions
    steps = max_steps // every_n

    plt.figure(figsize=(10, 6))

    # Plot the evolution of expected rewards
    plt.plot(range(every_n, max_steps + every_n, every_n), mean[:steps], label='Mean')

    # Fill between +- standard deviation
    plt.fill_between(range(every_n, max_steps + every_n, every_n), mean[:steps] - std[:steps], mean[:steps] + std[:steps], alpha=0.2)

    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    # Need to specify it is Inverted Double Pendulum if double is True
    plt.title('Mean Reward over Timesteps for DDPG (our implementation)' + (' with Inverted Double Pendulum' if double else ' with Inverted Pendulum'))
    plt.grid(True)
    plt.savefig('our_ddpg' + ('_double.png' if double else '_simple.png'))
    plt.close()

def main():
    print('Simple inverted pendulum')
    # Initialize the environment
    env = gym.make("InvertedPendulum-v4")
    noise_std = 0.5
    # Apply the algorithm
    _, _, all_mean_rewards = ddpg(env, n_episodes, n_transitions, noise_std, buffer_size, batch_size, tau, gamma, double=False)
    env.close()

    plot_rewards(all_mean_rewards, double=False)

    print('Double inverted pendulum')
    # Initialize the environment
    env = gym.make("InvertedDoublePendulum-v4")
    noise_std = 0.3
    # Apply the algorithm
    _, _, all_mean_rewards = ddpg(env, n_episodes, n_transitions, noise_std, buffer_size, batch_size, tau, gamma, double=True)
    env.close()

    plot_rewards(all_mean_rewards, double=True)

if __name__ == "__main__":
    main()

