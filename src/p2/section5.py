
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from section1 import *
from tqdm import tqdm
from section3 import make_video
from section4 import plot_q, plot_policy
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[64, 64]):
        super(QNetwork, self).__init__()
        layers = []
        input_dim = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.Tanh())
            input_dim = hidden_size
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class EpsilonGreedyPolicy:
    def __init__(self, model, start_epsilon=1.0, end_epsilon=0.01, decay_rate=0.995):
        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_rate = decay_rate
        self.model = model
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice([-4, 4])
        else:
            p, s = state
            if self.model(torch.tensor([p, s, 4], dtype=torch.float32).to(device)) >\
                  self.model(torch.tensor([p, s, -4], dtype=torch.float32).to(device)):
                return 4
            else:
                return -4

    def update_epsilon(self, epoch, total_epochs):
        # Exponential decay
        decay_factor = (self.end_epsilon / self.start_epsilon) ** (1 / total_epochs)
        self.epsilon = max(self.start_epsilon * (decay_factor ** epoch), self.end_epsilon)
            
class OptimalAgent:
    ''' Agent that selects the optimal action based on a given model. '''
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_action(self, state):
        p, s = state
        if self.model(torch.tensor([p, s, 4], dtype=torch.float32).to(self.device)) > self.model(torch.tensor([p, s, -4], dtype=torch.float32).to(self.device)):
            return 4
        else:
            return -4
            
class ReplayBuffer(Dataset):
    def __init__(self, capacity):
        super(ReplayBuffer, self).__init__()
        self.capacity = capacity
        self.counter = 0
        self.rew = 0
        self.rew2 = 0
        self.buffer = torch.empty((capacity, 7), dtype=torch.float32, device=device)

    def push(self, state, action, reward, next_state):
        if reward == 1 or reward == -1:
            done = 1
        else:
            done = 0
        
        if reward == 1:
            self.rew += 1
        elif reward == -1:
            self.rew2 += 1
        
        idx = self.counter % self.capacity
        p, s = state
        p_next, s_next = next_state
    
        self.buffer[idx] = torch.tensor([p, s, action, reward, p_next, s_next, done], dtype=torch.float32, device=device)
        if self.counter < self.capacity:
            self.counter += 1

    def sample(self, batch_size):
        idxs = torch.randint(0, min(self.counter, self.capacity), size=(batch_size,), device=device)
        return self.buffer[idxs]

    
    def __len__(self):
        return min(self.counter, self.capacity)

    def __getitem__(self, idx):
        return self.buffer[idx]



def parametric_q_learning(domain=Domain(), num_epochs=200, epsilon=.1, hidden_layers=[8, 16, 32, 16, 8], batch_size=8, buffer_size=100000, target_update_rate=1000):
    print(f"Using device: {device}")
    domain.reset()

    # Primary network
    model = QNetwork(input_size=3, output_size=1, hidden_sizes=hidden_layers).to(device)
    # Target network
    target_model = deepcopy(model)

    agent = EpsilonGreedyPolicy(model=model, start_epsilon=epsilon[0], end_epsilon=epsilon[1], decay_rate=epsilon[2])
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    replay_buffer = ReplayBuffer(buffer_size)

    model.train()

    for epoch in tqdm(range(num_epochs)):
        try:
            state = domain.get_state()
            action = agent.get_action(state)
            agent.update_epsilon(epoch, num_epochs)
            (p, s), a, r, (p_next, s_next) = domain.step(action, update=False)
            replay_buffer.push((p, s), a, r, (p_next, s_next))
            
            if r == 1 or r == -1:
                domain.reset()

            if replay_buffer.counter < batch_size:
                continue
            
            if epoch % target_update_rate == 0:
                target_model.load_state_dict(model.state_dict())

            batch = replay_buffer.sample(batch_size)
            X = batch[:, :3]
            y = batch[:, 3].unsqueeze(1)
            p_prime = batch[:, 5].unsqueeze(-1)
            s_prime = batch[:, 6].unsqueeze(-1)

            # Create a tensor of 4s, with the same batch size as p_prime and s_prime
            fours = torch.full_like(p_prime, 4)  # This creates a tensor of the same shape as p_prime but filled with 4
            minus_fours = torch.full_like(p_prime, -4)  # This creates a tensor of the same shape as p_prime but filled with 4

            # Concatenate along the second dimension to form a [N, 3] tensor
            z_pos = torch.cat([p_prime, s_prime, fours], dim=1).to(device)
            z_neg = torch.cat([p_prime, s_prime, minus_fours], dim=1).to(device)

            done = batch[:, -1].unsqueeze(1)

            optimizer.zero_grad()
            
            # Calculate the current Q values for the batch
            current_q_values = model(X)
            
            # Calculate the future Q values from next state for both actions
            future_q_values_pos = target_model(z_pos)
            future_q_values_neg = target_model(z_neg)
            
            # Take the max future Q value among the possible actions
            max_future_q_values, _ = torch.max(torch.cat((future_q_values_pos, future_q_values_neg), dim=1)\
                                            , dim=1, keepdim=True)
            
            # Calculate the expected Q values
            expected_q_values = y + (1 - done) * domain.discount * max_future_q_values
            # expected_q_values = y +  domain.discount * max_future_q_values
            
            # Calculate loss
            loss = criterion(current_q_values, expected_q_values.detach())
            
            # Perform backpropagation and an optimization step
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}", flush=True)
                print(f"The buffer is made up of {replay_buffer.counter} samples and of 1:{replay_buffer.rew}, -1:{replay_buffer.rew2}", flush=True)
                print(f"Current epsilon: {agent.epsilon}", flush=True)
        except KeyboardInterrupt:
            print("Interrupted, saving model...")
            break
    # save the model
    torch.save(model.state_dict(), "q_network.pth")  
    return model

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)


    batch_size = 8
    epsilon = [1, .1, .5]
    buffer_size = 100000
    num_epochs = 100000
    target_update_rate = 1000
    hidden_layers = [8, 16, 32, 16, 8]

    q_network = parametric_q_learning(num_epochs=num_epochs,\
             hidden_layers=hidden_layers, batch_size=batch_size, \
             epsilon=epsilon, buffer_size=buffer_size, target_update_rate=target_update_rate)
    
    domain = Domain()
    agent = OptimalAgent(q_network)
    domain.sample_initial_state()

    i = 0
    while True:
        if i > 3000:
            print("Reached 3000 steps, stopping.")
            break
        i += 1
        state = domain.get_state()
        action = agent.get_action(state)
        _, _, r, _ = domain.step(action)

        if r != 0:
            print(f"Reached terminal state after {i} steps.")
            break
    
    plot_policy(model=q_network, res=.01, options=('nn', 'nn', 'test'), path="./figures/section5")
    plot_q(model=q_network, res=.01, options=('nn', 'nn', 'test'), path="./figures/section5")

    make_video(domain.get_trajectory(), options=('nn', 'nn', 'test'))
