from section1 import Domain, MomentumAgent
from section2 import PolicyEstimator
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from section3 import make_video
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from itertools import product
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm

# 1. 1-step transitions
# 2. Create an input/output set with 1 step transitions and fitted q from last horizon
# 3. Train a model on the input/output set
def fitted_q_iteration(domain, alg, n_q, transitions, stopping=0):
    ''' Performs Fitted Q Iteration on the given domain and transitions with sklearn. '''
    X, y, z = transitions # X -> (p, s, a), y -> r, z -> (p_prime, s_prime)
    z_pos = np.hstack((z, np.full((z.shape[0], 1), 4)))
    z_neg = np.hstack((z, np.full((z.shape[0], 1), -4)))
    error = []

    if alg == 'linear':
        model = LinearRegression()

    elif alg == 'trees':
        model = ExtraTreesRegressor(n_estimators=50, min_samples_leaf=2, random_state=42)
        # parameters chosen based on paper.
    elif alg == 'nn':
        return nn_fitted_q_iteration(domain, alg, 200, transitions, stopping=stopping)
    else :
        raise ValueError("Invalid algorithm for learning.")
    
    model.fit(X, y)
    for _ in tqdm(range(n_q-1)):
        concat = np.column_stack((model.predict(z_pos), model.predict(z_neg)))
        out = y + domain.discount * np.max(concat, axis=1)

        l_pred = model.predict(X)
        model.fit(X, out)
        r_pred = model.predict(X)

        e = mse(l_pred, r_pred)
        error.append(e)
        if stopping and e < 1e-4:
            break
        
    return model, error
    
class QNetwork(nn.Module):
    ''' Neural network for Q-learning '''
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, output_size),
        )
        
    def forward(self, x):
        return self.network(x) 

def nn_fitted_q_iteration(domain, alg, n_q, transitions, stopping=0, batch_size=256):
    ''' Performs Fitted Q Iteration on the given domain and transitions with a neural network. '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y, z = transitions
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
    z_pos = torch.tensor(np.hstack((z, np.full((z.shape[0], 1), 4))), dtype=torch.float32).to(device)
    z_neg = torch.tensor(np.hstack((z, np.full((z.shape[0], 1), -4))), dtype=torch.float32).to(device)

    # Creating a dataset and dataloader for mini-batch training
    dataset = TensorDataset(X, y, z_pos, z_neg)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = QNetwork(input_size=X.shape[1], output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    error = []

    model.train()
    
    for X_batch, y_batch, z_pos_batch, z_neg_batch in dataloader:
        X_batch, y_batch, z_pos_batch, z_neg_batch =  X_batch.to(device), \
            y_batch.to(device), z_pos_batch.to(device), z_neg_batch.to(device)
        
        optimizer.zero_grad()
        concat = torch.max(model(z_pos_batch), model(z_neg_batch))
        l_pred = model(X_batch)
        loss = criterion(l_pred, y_batch)
        loss.backward()
        optimizer.step()
    

    for _ in tqdm(range(n_q-1)):
        with torch.no_grad():
            l_pred = model(X)
        for X_batch, y_batch, z_pos_batch, z_neg_batch in dataloader:
            X_batch, y_batch, z_pos_batch, z_neg_batch =  X_batch.to(device), \
                y_batch.to(device), z_pos_batch.to(device), z_neg_batch.to(device)
            
            optimizer.zero_grad()
            concat = torch.max(model(z_pos_batch), model(z_neg_batch))
            out = y_batch + domain.discount * concat
            l_pred_batch = model(X_batch)
            loss = criterion(l_pred_batch, out)
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            r_pred = model(X)
            e = criterion(l_pred, r_pred).cpu().detach().numpy()  # Consider revising for batch logic
            error.append(e)
            if stopping and e < 3e-4:
                break
                
    return model, error

def mse(old_pred, new_pred):
    ''' Computes MSE between Q_N and Q_(N-1) '''
    return np.mean((old_pred-new_pred)**2)
    
def sample_state(strategy="uniform"):
    if strategy == "uniform":
        p = np.random.uniform(-1, 1.00001)
        while p > 1:
            p = np.random.uniform(-1, 1.00001)
        s = np.random.uniform(-3, 3.00001)
        while s > 3:
            s = np.random.uniform(-3, 3.00001)
        return p, s
    else:
        raise ValueError("Invalid strategy for sampling p.")

def traj_step(domain=Domain(), strategy='uniform', X=[], y=[], z=[]):
    ''' Generate a trajectory step and append it to the dataset using a given strategy.'''
    p, s = sample_state()

    domain.reset()
    domain.set_state(p, s)

    a = np.random.choice([4, -4])
    (p, s), a, r, (p_prime, s_prime) = domain.step(a)
    X.append((p, s, a))
    y.append(r)
    z.append((p_prime, s_prime))
    return X, y, z

def get_set(domain=Domain(), mode='randn', n_iter=int(1e4)):
    ''' Generate a dataset of transitions using a given mode. 
    mode: 'randn' for random sampling, 'episodic' for episodic sampling (further explained in the report).'''
    domain.sample_initial_state()
    if mode == 'randn':
        X, y, z = traj_step(domain=domain)
        for _ in tqdm(range(n_iter)):
            X, y, z = traj_step(domain=domain, strategy='uniform', X=X, y=y)
        return np.array(X), np.array(y), np.array(z)
    elif mode == 'episodic':
        n_episodes = n_iter
        X, y, z = [], [], []
        for _ in tqdm(range(n_episodes)):
            r = 0
            domain.sample_initial_state()
            while r == 0:
                (p, s), a, r, (p_next, s_next) = domain.step(np.random.choice([4, -4]), update=False)
                X.append((p, s, a))
                y.append(r)
                z.append((p_next, s_next))
            domain.reset()
        print(np.unique(np.array(y), return_counts=True))   
        print(f"Generated {n_episodes} episodes. and X shape: {np.array(X).shape}")
        return np.array(X), np.array(y), np.array(z)
    elif mode == 'uni-episodic':
        n_episodes = n_iter
        X, y, z = [], [], []
        for _ in tqdm(range(n_episodes)):
            r = 0
            domain.sample_initial_state()
            sample_p = np.random.uniform(-1, 1.00001)
            sample_s = np.random.uniform(-3, 3.00001)
            domain.set_state(sample_p, sample_s)
            while r == 0:
                (p, s), a, r, (p_next, s_next) = domain.step(np.random.choice([4, -4]), update=False)
                X.append((p, s, a))
                y.append(r)
                z.append((p_next, s_next))
            domain.reset()
        print(np.unique(np.array(y), return_counts=True))   
        print(f"Generated {n_episodes} episodes. and X shape: {np.array(X).shape}")
        return np.array(X), np.array(y), np.array(z)
    else:
        raise ValueError("Invalid mode for generating dataset.")


def q_eval(p, s, a, model, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    ''' Evaluate the Q function for a given state-action pair using a given model.'''
    psa = np.stack((p, s, np.full(p.shape, a)), axis=-1).reshape(-1, 3)

    if isinstance(model, torch.nn.Module):
        psa_tensor = torch.tensor(psa, dtype=torch.float32).to(device)
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Temporarily set all the requires_grad flag to false
            output = model(psa_tensor)
        return output.cpu().numpy()
    else:
        return model.predict(psa)

def plot_q(model, res, options, path = "../../figures/project2/section4"):
    '''Plot the estimation of Q_N'''
    alg, traj, stop = options
    options = alg + '_' + traj + '_' + 'stop:' + str(stop)

    p_values = np.arange(-1, 1 + res, res)  # Range of values considered for p
    s_values = np.arange(-3, 3 + res, res)  # Range of values considered for s

    X, Y = np.meshgrid(p_values, s_values)  # Create the grid

    for a in [4, -4]:
        Z = q_eval(X, Y, a, model)  # Evaluate the function for all (p,s) combinations
        Z = Z.reshape(X.shape)  # Reshape Z back to the grid shape

        plt.figure()  # Create a new figure
        plt.imshow(Z, cmap='coolwarm', origin='lower',
                   extent=[p_values.min(), p_values.max(), s_values.min(), s_values.max()])
        plt.colorbar(label='$\hat{Q}_N$')
        plt.xlabel('p')
        plt.ylabel('s')
        # plt.title(f'$\hat{{Q}}_N$ value for action {a}') # need to be modified so that it states options
        plt.axis('tight')
        plt.savefig(path+f'/Q_N_action_{a}_{options}.png')   # Save the plot
        plt.close()

class OptimalAgent:
    ''' Agent that selects the optimal action based on a given model. '''
    def __init__(self, model, alg):
        self.model = model
        self.alg = alg # specifies which learning algorithm was used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_action(self, state):
        p, s = state
        if self.alg == 'nn':
            if self.model(torch.tensor([p, s, 4], dtype=torch.float32).to(self.device)) > self.model(torch.tensor([p, s, -4], dtype=torch.float32).to(self.device)):
                return 4
            else:
                return -4
        else:
            if self.model.predict(np.array([(p,s,4)])) > self.model.predict(np.array([(p,s,-4)])):
                return 4
            else:
                return -4
            
def plot_policy(model, res, options, path="../../figures/project2/section4"):
    '''Plot an agent's policy using a given resolution'''
    alg, traj, stop = options
    options = alg + '_' + traj + '_' + 'stop:' + str(stop)

    p_values = np.arange(-1, 1 + res, res)
    s_values = np.arange(-3, 3 + res, res)
    
    P, S = np.meshgrid(p_values, s_values, indexing='ij')

    values_grid_1 = np.dstack((P, S, 4*np.ones_like(P)))
    values_grid_2 = np.dstack((P, S, -4*np.ones_like(P)))

    values_grid_1 = values_grid_1.reshape((len(p_values)*len(s_values), 3))
    values_grid_2 = values_grid_2.reshape((len(p_values)*len(s_values), 3))
    
    pred_1 = model.predict(values_grid_1) if alg != 'nn' else model(torch.tensor(values_grid_1, dtype=torch.float32).to(device)).cpu().detach().numpy()
    pred_2 = model.predict(values_grid_2) if alg != 'nn' else model(torch.tensor(values_grid_2, dtype=torch.float32).to(device)).cpu().detach().numpy()
    
    pred_1 = pred_1.reshape(len(p_values),len(s_values))
    pred_2 = pred_2.reshape(len(p_values),len(s_values))

    all_pred = np.stack((pred_1, pred_2), axis=-1)

    policy_grid = np.argmax(all_pred, axis = 2)
    
    indices_pos = np.argwhere(policy_grid == 1)
    indices_neg = np.argwhere(policy_grid == 0)

    plt.scatter((indices_pos[:, 0] - len(p_values)/2)*res, (indices_pos[:, 1] - len(s_values)/2)*res, color='blue', label='4')
    plt.scatter((indices_neg[:, 0] - len(p_values)/2)*res, (indices_neg[:, 1] - len(s_values)/2)*res, color='red', label='-4')

    plt.xlabel('p')
    plt.ylabel('s')

    plt.xlim(-1,1)
    plt.ylim(-3,3)

    plt.legend()


    plt.grid(True)
    
    plt.savefig(path+f'/est_opt_policy_{options}.png') 
    plt.close()
    print('policy plotted')

def plot_e(traj, alg, stop, error, path="../../figures/project2/section4"):
    '''Plot the evolution of the MSE between Q_N and Q_(N-1)'''
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(error)), error)
    # plt.title('MSE between Q_N and Q_(N-1)')
    plt.xlabel('N')
    plt.ylabel('MSE')
    plt.grid(True, which="both", ls="--")
    plt.savefig(path+f'mse_{traj}_{alg}_{stop}.png')
    plt.close()

if __name__ == "__main__":
    ''' In this main, we will run the experiments for section 4.
    We will consider the different trajectories and algorithms and stopping criteria.
    We will also plot the Q functions and policies for each model obtained as well as the evolution of the MSE and gif of the trajectory.'''
    domain = Domain()

    all_traj = ['randn', 'episodic']
    all_alg = ['linear', 'trees', 'nn']
    all_stop = [1, 0]
    all_res = [.01]

    evol_avg_return = {}

    print('Starting section 4...')
    for traj in all_traj:
        iters = 100000 if traj == 'randn' else 1000
        # iters = 10000 if traj == 'randn' else 10 # TO TEST

        print(f'Fetching {traj} set...')
        X, y, z = get_set(mode=traj, n_iter=int(iters))
        for alg, stop, res in product(all_alg, all_stop, all_res):
            print(f'Running with {alg} algorithm and stopping is : {bool(stop)}')
            options = (alg, traj, stop)


            N = 50 # might change if we change stopping criterion
            print('Fitting...')
            model, error = fitted_q_iteration(domain, alg, N, (X, y, z), stopping=stop)

            print("Plotting...")
            plot_e(traj, alg, stop, error, path='figures/section4/error/')
            plot_q(model, res, options=options, path='figures/section4/qfuncs/')
            plot_policy(model, res, options=options, path='figures/section4/policies/')

            opt_agent = OptimalAgent(model, alg=alg)

            domain.reset()
            r = 0
            i = 0
            while r == 0:
                i += 1
                if i > 2000:
                    print('Infinite loop detected. Stopping simulation.')
                    break
                state = domain.get_state()
                action = opt_agent.get_action(state)
                _, _, r, _ = domain.step(action)
            make_video(domain.get_trajectory(), options=options, path='figures/section4/gifs')

            policy_est = PolicyEstimator(domain, opt_agent)

            n_initials = 50 
            N = 300 
            print(f"Applying monte carlo for Expected return {n_initials} times with horizon {N}...")
            all_returns = policy_est.policy_return(N, n_initials) # Get all the estimated expected returns
            policy_est.plot_return(all_returns, filename="_"+alg+traj+str(stop), path="figures/section4/ereturns") # Plot the returns for the 50 initial states

            evol_avg_return[options] = all_returns

            print(f'Estimated expected return of policy {options}: {np.mean(all_returns[:,-1])}') # Print the average so that we also have it numerically
    
    plt.figure(figsize=(20, 12  ))

    for options, returns in evol_avg_return.items():
        N = returns.shape[1]  # Horizon length
        mean_returns = np.mean(returns, axis=0)  # Mean over simulations
        plt.plot(range(1, N + 1), mean_returns, label=f'Options: {options}')

    plt.xlabel('N')
    plt.ylabel('Mean expected return')
    plt.xlim((1, N + 1))
    plt.legend()
    plt.grid(True, which="both", ls="--")

    plt.savefig('figures/section4/ereturns/mean_exp_return_FINAL.png')
    plt.close()

