
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
from collections import deque
from section1 import *
import tqdm as tqdm

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
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
    
    def get_action(self, model, state):
        if random.random() < self.epsilon:
            return random.choice[-4, 4]
        else:
            p, s = state
            if model(torch.tensor([p, s, 4], dtype=torch.float32).to(self.device)) >\
                  model(torch.tensor([p, s, -4], dtype=torch.float32).to(self.device)):
                return 4
            else:
                return -4
        


def parametric_q_learning_step(num_epochs=200, hidden_layers=[8, 16, 32, 16, 8]):
    print(f"Using device: {device}")
    
    model = QNetwork(input_size=3, output_size=1, hidden_sizes=hidden_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()

    # num_epochs -= 1
    # for X_batch, y_batch, z_pos_batch, z_neg_batch in dataloader:
        
    #     optimizer.zero_grad()
    #     concat = torch.max(model(z_pos_batch), model(z_neg_batch))
    #     out = y_batch
    #     loss = criterion(out, )
    #     loss.backward()
    #     optimizer.step()

    for epoch in tqdm(range(num_epochs)):
        (p, s), a, r, (p_next, s_next) = domain.step(epolicy.get_action(model, domain.state))
        X = torch.tensor([p, s, a], dtype=torch.float32).to(device)
        y = torch.tensor([r], dtype=torch.float32).to(device)
        z_pos = torch.tensor([p_next, s_next, 4], dtype=torch.float32).to(device)
        z_neg = torch.tensor([p_next, s_next, -4], dtype=torch.float32).to(device)
        
        optimizer.zero_grad()
        
        # Calculate the current Q values for the batch
        current_q_values = model(X)
        
        # Calculate the future Q values from next state for both actions
        future_q_values_pos = model(z_pos)
        future_q_values_neg = model(z_neg)
        
        # Take the max future Q value among the possible actions
        max_future_q_values, _ = torch.max(torch.cat((future_q_values_pos, future_q_values_neg), dim=1)\
                                           , dim=1, keepdim=True)
        
        # Calculate the expected Q values
        expected_q_values = y + domain.discount * max_future_q_values
        
        # Calculate loss
        loss = criterion(current_q_values, expected_q_values.detach())
        
        # Perform backpropagation and an optimization step
        loss.backward()
        optimizer.step()

                
    return model

if __name__ == "__main__":
    batch_size = 8

    domain = Domain()
    epolicy = EpsilonGreedyPolicy()
    model = QNetwork(input_size=3, output_size=1, hidden_sizes=[8, 16, 32, 16, 8]).to(device)
