from section1 import Domain, MomentumAgent
from section2 import PolicyEstimator
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

## TO DELETE
from tqdm import tqdm
import pandas as pd

# 1. 1-step transitions
# 2. Create an input/output set with 1 step transitions and fitted q from last horizon
# 3. Train a model on the input/output set
def fitted_q_iteration(domain, alg, n_q, transitions, stopping=0):
    X, y, z = transitions # X -> (p, s, a), y -> r, z -> (p_prime, s_prime)
    z_pos = np.hstack((z, np.full((z.shape[0], 1), 4)))
    z_neg = np.hstack((z, np.full((z.shape[0], 1), -4)))
    error = []

    if alg == 'linear':
        model = LinearRegression()

    elif alg == 'trees':
        model = ExtraTreesRegressor(n_estimators= 50, max_features = 2, min_samples_leaf = 2)

    model.fit(X, y)
    for _ in range(n_q-1):
        concat = np.column_stack((model.predict(z_pos), model.predict(z_neg)))
        out = y + domain.discount * np.max(concat, axis=1)

        l_pred = model.predict(X)
        model.fit(X, out)
        r_pred = model.predict(X)

        e = mse(l_pred, r_pred)
        error.append(e)
        if stopping and e < 1e-3:
            break
        
    return model, error

def mse(old_pred, new_pred):
    ''' Computes MSE between Q_N and Q_(N-1) '''
    return np.mean((old_pred-new_pred)**2)
    
def sample_state(strategy="uniform"):
    if strategy == "uniform":
        return np.random.uniform(-1, 1.00001), np.random.uniform(-3, 3.00001)
    else:
        raise ValueError("Invalid strategy for sampling p.")

def traj_step(domain=Domain(), strategy='uniform', X=[], y=[], z=[]):
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
            while not domain.get_rflag():
                (p, s), a, r, (p_next, s_next) = domain.step(np.random.choice([4, -4]))
                X.append((p, s, a))
                y.append(r)
                z.append((p_next, s_next))
            domain.reset()
        print(np.unique(np.array(y), return_counts=True))   
        print(f"Generated {n_episodes} episodes. and X shape: {np.array(X).shape}")
        return np.array(X), np.array(y), np.array(z)
    ################## TO DELETE ##################
    elif mode == 'csv':
        df = pd.read_csv(f"data/traj_len_{n_iter}.csv")
        X = df[['p', 's', 'a']]
        y = df['r']
        z = df[['p_prime', 's_prime']]
        return X.to_numpy(), y.to_numpy(), z.to_numpy()
    ###############################################
    else:
        raise ValueError("Invalid mode for generating dataset.")


def q_eval(p, s, a, model):
    # Reformatted to create a 2D array suitable for model.predict
    # Stack p, s, and a arrays along the last axis and reshape into (-1, 3) for model input
    psa = np.stack((p, s, np.full(p.shape, a)), axis=-1).reshape(-1, 3)
    print(psa)
    return model.predict(psa)

def plot_q(model, options ="", path = "../../figures/project2/section2"):
    '''Plot the estimation of Q_N'''
    p_values = np.arange(-1, 1.01, 0.01)  # Range of values considered for p
    s_values = np.arange(-3, 3.01, 0.01)  # Range of values considered for s

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
        plt.title(f'$\hat{{Q}}_N$ value for action {a}')
        plt.axis('tight')
        plt.savefig(path+f'/Q_N_action_{a}_{options}.png')   # Save the plot

class OptimalAgent: # do we need another agent specifically for NN or is that ok?
    def __init__(self, model, alg):
        self.model = model
        self.alg = alg # specifies which learning algorithm was used

    def get_action(self, state):
        p, s = state
        if self.alg == 'nn':
            return 0 # what do we do in neural net case
        else:
            if model.predict(np.array([(p,s,4)])) > model.predict(np.array([(p,s,-4)])):
                return 4
            else:
                return -4
            
def plot_policy(agent, res, options = "", path="../../figures/project2/section2"):
    '''Plot an agent's policy using a given resolution'''
    p_values = np.arange(-1, 1 + res, res)
    s_values = np.arange(-3, 3 + res, res)

    policy_grid = np.empty((len(p_values), len(s_values)))

    for i in range(len(p_values)):
        for j in range(len(s_values)):
            policy_grid[i,j] = agent.get_action((p_values[i],s_values[j]))

    indices_pos = np.argwhere(policy_grid == 4)
    indices_neg = np.argwhere(policy_grid == -4)

    plt.scatter((indices_pos[:, 0] - len(p_values)/2)*res, (indices_pos[:, 1] - len(s_values)/2)*res, color='blue', label='4')
    plt.scatter((indices_neg[:, 0] - len(p_values)/2)*res, (indices_neg[:, 1] - len(s_values)/2)*res, color='red', label='-4')

    plt.xlabel('p')
    plt.ylabel('s')

    plt.xlim(-1,1)
    plt.ylim(-3,3)

    plt.legend()

    plt.grid(True) # Not sure the grid makes sense
    
    # options should be more specific: learning algorithm, stopping criterion, trajectory used    
    plt.savefig(path+f'/est_opt_policy_{options}.png') 

if __name__ == "__main__":
    domain = Domain() # need to make sure that by putting it here, we don't miss a domain reset anywhere

    # all_traj = ['randn', 'episodic']
    # all_alg = ['linear', 'trees', 'nn']
    # all_stop = [0, 1]

    traj = 'randn' # need to use a for loop
    X, y, z = get_set(mode=traj, n_iter=int(1000))

    N = 50 # might change if we change stopping criterion
    alg = 'linear' # need to use a for loop
    stop = 0 # need to use a for loop
    model, error = fitted_q_iteration(domain, alg, N, (X, y, z), stopping = stop)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(error)), error)
    plt.title('MSE between Q_N and Q_(N-1)')
    plt.xlabel('N')
    plt.ylabel('MSE')
    plt.grid(True, which="both", ls="--")
    plt.savefig('mse.png')

    options = alg + '_' + traj + '_' + 'stop' + str(stop)
    plot_q(model, options=options)

    # once we have final model
    opt_agent = OptimalAgent(model, alg = 'linear')

    res = 0.01 # resolution for the plot
    plot_policy(opt_agent, res, options = options)

    # Estimating the return of the policy:
    policy_est = PolicyEstimator(domain, opt_agent)

    n_initials = 50 # Number of initial states
    N = 5000 # Horizon

    # In this case, wouldn't it be more interesting to have a single plot with the 
    # estimated expected return (averaged over all initial states basically) of all models obtained?

    # if we just do one plot per model:
    all_returns = policy_est.policy_return(N, n_initials) # Get all the estimated expected returns

    policy_est.plot_return(all_returns, filename="_"+options) # Plot the returns for the 50 initial states

    # If we want to do one plot for all models, can just store for each model:
    evol_avg_return = np.mean(all_returns, axis = 0) # Then can just adapt the policy_est.plot_return function very easily

    print(f'Estimated expected return of policy {options}: {np.mean(all_returns[:,-1])}') # Print the average so that we also have it numerically




