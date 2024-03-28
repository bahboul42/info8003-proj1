from section1 import Domain, MomentumAgent
from section2 import PolicyEstimator
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from section3 import make_video
import torch
from itertools import product
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
        model = ExtraTreesRegressor(n_estimators=50, min_samples_leaf=2, random_state=42)
        # parameters chosen based on paper. note: they say k = input size, which is default I think : yes
    elif alg == 'nn':
        return nn_fitted_q_iteration(domain, alg, n_q, transitions, stopping=0)
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

def nn_fitted_q_iteration(domain, alg, n_q, transitions, stopping=0):
    pass

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
            domain.set_state(-.5, 0)
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


def q_eval(p, s, a, model):
    # Reformatted to create a 2D array suitable for model.predict
    # Stack p, s, and a arrays along the last axis and reshape into (-1, 3) for model input
    psa = np.stack((p, s, np.full(p.shape, a)), axis=-1).reshape(-1, 3)
    # print(psa)
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
    
    pred_1 = model.predict(values_grid_1)
    pred_2 = model.predict(values_grid_2)
    
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

    # plt.title(f'Optimal policy for {options}')

    plt.grid(True) # Not sure the grid makes sense
    
    # options should be more specific: learning algorithm, stopping criterion, trajectory used    
    plt.savefig(path+f'/est_opt_policy_{options}.png') 
    plt.close()
    print('policy plotted')

def plot_e(traj, alg, stop, error, path="../../figures/project2/section4"):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(error)), error)
    # plt.title('MSE between Q_N and Q_(N-1)')
    plt.xlabel('N')
    plt.ylabel('MSE')
    plt.grid(True, which="both", ls="--")
    plt.savefig(path+f'mse_{traj}_{alg}_{stop}.png')
    plt.close()

if __name__ == "__main__":
    np.random.seed(0)
    domain = Domain()

    # all_traj = ['randn', 'episodic']
    all_traj = ['randn']
    all_alg = ['linear', 'trees']
    all_stop = [0, 1]
    all_res = [.01]

    evol_avg_return = {}

    print('Starting section 4...')
    for traj in all_traj:
        iters = 10000 if traj == 'randn' else 1000
        print(f'Fetching {traj} set...')
        X, y, z = get_set(mode=traj, n_iter=int(iters))
        for alg, stop, res in product(all_alg, all_stop, all_res):
            print(f'Running with {alg} algorithm and stopping is : {bool(stop)}')
            options = (alg, traj, stop)


            N = 50 # might change if we change stopping criterion
            print('Fitting...')
            model, error = fitted_q_iteration(domain, alg, N, (X, y, z), stopping=stop)

            print("Plotting...")
            plot_e(traj, alg, stop, error, path='figures/section4/error')
            plot_q(model, res, options=options, path='figures/section4/qfuncs')
            plot_policy(model, res, options=options, path='figures/section4/policies')

            opt_agent = OptimalAgent(model, alg=alg)

            domain.reset()
            r = 0
            while r == 0:
                state = domain.get_state()
                action = opt_agent.get_action(state)
                _, _, r, _ = domain.step(action)
            make_video(domain.get_trajectory(), options=options, path='figures/section4/gifs')

            policy_est = PolicyEstimator(domain, opt_agent)

            n_initials = 50 
            N = 300 
            print(f"Applying monte carlo for Expected return {n_initials} times with horizon {N}...")
            # In this case, wouldn't it be more interesting to have a single plot with the 
            # estimated expected return (averaged over all initial states basically) of all models obtained?

            all_returns = policy_est.policy_return(N, n_initials) # Get all the estimated expected returns
            policy_est.plot_return(all_returns, filename="_"+alg+traj+str(stop), path="figures/section4/ereturns") # Plot the returns for the 50 initial states

            evol_avg_return[options] = all_returns

            print(f'Estimated expected return of policy {options}: {np.mean(all_returns[:,-1])}') # Print the average so that we also have it numerically
    
    policy_est.plot_mean_returns(evol_avg_return, filename="_FINAL", path="figures/section4/ereturns")

