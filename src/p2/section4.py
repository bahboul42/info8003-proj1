from section1 import Domain, MomentumAgent
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

## TO DELETE
from tqdm import tqdm
import pandas as pd

# 1. 1-step transitions
# 2. Create an input/output set with 1 step transitions and fitted q from last horizon
# 3. Train a model on the input/output set
def fitted_q_iteration(domain, model, n_q, transitions, stopping=0):
    X, y, z = transitions # X -> (p, s, a), y -> r, z -> (p_prime, s_prime)
    z_pos = np.hstack((z, np.full((z.shape[0], 1), 4)))
    z_neg = np.hstack((z, np.full((z.shape[0], 1), -4)))
    error = []

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
    return model.predict(psa)

def plot_q(model):
    p_values = np.arange(-1, 1, 0.01)  # Range of values considered for p
    s_values = np.arange(-3, 3, 0.01)  # Range of values considered for s

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
        plt.show()  # Show the plot

if __name__ == "__main__":
    
    X, y, z = get_set(mode='episodic', n_iter=int(1000))

    domain = Domain()
    model = LinearRegression()

    N = 150
    model, error = fitted_q_iteration(domain, model, N, (X, y, z))

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(error)), error)
    plt.title('MSE between Q_N and Q_(N-1)')
    plt.xlabel('N')
    plt.ylabel('MSE')
    plt.grid(True, which="both", ls="--")
    plt.savefig('mse.png')


    plot_q(model)


