from section1 import Domain
from itertools import product
import numpy as np
import sys
from functools import lru_cache
from section3 import MDP
import matplotlib.pyplot as plt

class OfflineQ:
    def __init__(self, domain, alpha_k):
        self.domain = domain
        self.alpha_k = alpha_k

    def rand_policy(self):
        return self.domain.actions[np.random.choice(len(self.domain.actions))]

    # Only modif from get_sequence from MDPEstimator class is that I also put next_state in the tuples
    def generate_sequence(self, sequence_len, s0, type="det"):
        """Generate a sequence of states, actions, and rewards for Deterministic domain"""
        sequence = []

        curr_state = s0
        # Generate a sequence of states, actions, and rewards for the given sequence length
        for _ in range(sequence_len):
            state = curr_state
            action = self.rand_policy() # get random action
            if type == "det":
                reward = self.domain.det_reward(state, action) # get reward from action and current state
                next_state = self.domain.det_dyn(state, action) # get next state from action and current state
            else:
                reward = self.domain.sto_reward(state, action) # get reward from action and current state
                next_state = self.domain.sto_dyn(state, action) # get next state from action and current state
            sequence.append((state, action, reward, next_state)) # append to sequence
            curr_state = next_state # update current state to next state
        return sequence
    
    def update_q_hat(self, q_hat_n, transition):
        s, a, r, s_prime = transition
        new_trans = (1-self.alpha_k)*q_hat_n[(s,a)] + self.alpha_k*(r + self.domain.discount*np.max(np.array([q_hat_n[(s_prime,a_prime)] for a_prime in self.domain.actions])))
        q_hat_n[(s, a)] = new_trans

    def opt_policy(self, q_hat):
        mu_N = {}
        for i, j in product(range(self.domain.n), range(self.domain.m)):
            s = (i,j)
            mu_N[s] = self.domain.actions[np.argmax(np.array([q_hat[s,a] for a in self.domain.actions]))]
            # mu_N[s] = np.argmax(np.array([q_hat[s,a] for a in self.domain.actions]))
            
        return mu_N
    
    def compute_inf_norm(self, pred, true):
        """Compute the infinity norm of the difference between the true and estimated parameters."""
        max_diff = 0
        for i, j, a in product(range(self.domain.n), range(self.domain.m), self.domain.actions):
            s = (i,j)
            true_r = true.get((s,a), 0)
            estimated_r = pred.get((s,a), 0)
            diff = np.abs(true_r - estimated_r)
            if diff > max_diff:
                max_diff = diff
        
        return max_diff
    
    def print_policy(self, policy, title='Policy'):
        X, Y = np.meshgrid(np.arange(0.5, 5, 1), np.arange(0.5, 5, 1))
        U = np.zeros_like(X, dtype=float)
        V = np.zeros_like(Y, dtype=float)

        # Setting up the arrows based on the policy
        for i in range(5):
            for j in range(5):
                action = policy[(i,j)]
                U[4 - i, j] = action[1]  # dx
                V[4 - i, j] = -action[0]  # dy   


        plt.figure(figsize=(10, 10))
        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=3, width=0.003, pivot='mid')
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks(range(self.domain.m))
        plt.yticks(range(self.domain.n))
        plt.grid()

        # Adding the rewards text
        for i in range(5):
            for j in range(5):
                plt.text(j + 0.7, 5 - i - 0.7, self.domain.g[i, j], ha='center', va='top', color='red')

        plt.title(title)
        plt.show()
    
class OnlineQ:
    def __init__(self, domain, alpha, epsilon, mdp):
        self.domain = domain
        self.alpha = alpha
        self.epsilon = epsilon
        self.mdp = mdp
        self.det_mu_star = {}
        self.sto_mu_star = {}
        self.det_J_mu_star = np.empty(self.domain.g.size) # {} array easier for infinite norm
        self.sto_J_mu_star = np.empty(self.domain.g.size) # {}

    def mu_star(self, type = "det"):
        if type == "det":
            self.mdp.det_policy.cache_clear() # Clearing the cache
            self.mdp.det_Q_N.cache_clear() # Clearing the cache
            opt_policy = self.mdp.det_policy(10)
        else:
            self.mdp.det_policy.cache_clear() # Clearing the cache
            self.mdp.det_Q_N.cache_clear() # Clearing the cache
            opt_policy = self.mdp.sto_policy(10)

        policy = [self.domain.actions[i] for i in opt_policy]
        for i, j in product(range(self.domain.n), range(self.domain.m)):
            s = (i,j)
            if type == "det":
                self.det_mu_star[s] = policy[i * self.domain.n + j]
            else:
                self.sto_mu_star[s] = policy[i * self.domain.n + j]

    def j_star(self, type = "det"):
        if type == "det":
            policy = [self.det_mu_star[(i,j)] for i, j in product(range(domain.n), range(domain.m))]
            for i, j in product(range(self.domain.n), range(self.domain.m)):
                s = (i,j)
                self.det_J_mu_star[i * self.domain.n + j] = self.j_func(s, tuple(policy), 10**4, type = "det")

        else:
            policy = [self.sto_mu_star[(i,j)] for i, j in product(range(domain.n), range(domain.m))]
            for i, j in product(range(self.domain.n), range(self.domain.m)):
                s = (i,j)
                self.sto_J_mu_star[i * self.domain.n + j] = self.j_func(s, tuple(policy), 10**4, type = "sto")
    

    # J function for the deterministic domain
    @lru_cache(maxsize=None)
    def j_func(self, state, policy, N, type = "det"):
        if N == 0:
            return 0
        else:
            j = 0
            state_index = state[0] * domain.n + state[1]
            action = policy[state_index]
            if type == "det":
                j = self.domain.det_reward(state, action) + self.domain.discount * self.j_func(domain.dynamic(state, action), policy, N-1, type = "det")
            else:
                j = 0.5 * (self.domain.det_reward(state, action) + self.domain.discount * self.j_func(domain.dynamic(state, action), policy, N-1, type = "sto"))
                j += 0.5 * (self.domain.g[0, 0] + self.domain.discount * self.j_func((0, 0), policy, N-1, type = "sto"))
            return j
    
    def sim_episodes(self, s0, n_transitions, n_episodes, type = "det", protocol = 1):
        differences = np.zeros(n_episodes)
        differences_l2 = np.zeros(n_episodes)

        # Initializing q_hat
        q_hat = {}
        for i, j, a in product(range(self.domain.n), range(self.domain.m), self.domain.actions):
            s = (i,j)
            q_hat[(s,a)] = 0
        
        for ep in range(n_episodes):
            self.q_learning(q_hat, s0, n_transitions, type, protocol)
            mu_Q_hat = self.mu_q_hat(q_hat)
            policy = [mu_Q_hat[(i,j)] for i, j in product(range(self.domain.n), range(self.domain.m))]
            j_mu_hat = np.array([self.j_func((i,j), tuple(policy), 10**4, type) for i, j in product(range(self.domain.n), range(self.domain.m))])

            if type == "det":
                differences[ep] = np.max(np.abs(j_mu_hat - self.det_J_mu_star))
                differences_l2[ep] = np.linalg.norm(j_mu_hat - self.det_J_mu_star)
            else:
                differences[ep] = np.max(np.abs(j_mu_hat - self.sto_J_mu_star))
                differences_l2[ep] = np.linalg.norm(j_mu_hat - self.sto_J_mu_star)

        return differences, differences_l2

    
    # Policy given a q_function
    def mu_q_hat(self, q_hat):
        mu_q_hat = {}
        for i, j in product(range(self.domain.n), range(self.domain.m)):
            s = (i,j)
            mu_q_hat[s] = self.domain.actions[np.argmax(np.array([q_hat[s,a] for a in self.domain.actions]))]
            # mu_N[s] = np.argmax(np.array([q_hat[s,a] for a in self.domain.actions]))
        return mu_q_hat
    
    # Online Q-learning
    def q_learning(self, q_hat, s0, n_transitions, type = "det", protocol = 1):
        # Initialize alpha
        alpha_t = self.alpha

        # Starting state
        curr_state = s0

        # Replay buffer
        if protocol == 3:
            transitions = []

        for _ in range(n_transitions):
            s = curr_state
            a = self.eps_greedy(q_hat, s)

            if type == "det":
                r = self.domain.det_reward(s, a) # get reward from action and current state
                s_prime = self.domain.det_dyn(s, a) # get next state from action and current state
            else:
                r = self.domain.sto_reward(s, a) # get reward from action and current state
                s_prime = self.domain.sto_dyn(s, a) # get next state from action and current state

            # update q hat
            if protocol == 3:
                transitions.append((s,a,r,s_prime))
                for _ in range(10):
                    s_up, a_up, r_up, s_prime_up = transitions[np.random.choice(len(transitions))]
                    buffer = (1-alpha_t)*q_hat[(s_up,a_up)] + alpha_t*\
                        (r_up + self.domain.discount*np.max(np.array([q_hat[(s_prime_up,a_prime)] for a_prime in self.domain.actions])))
                    q_hat[(s_up,a_up)] = buffer
            else:
                buffer = (1-alpha_t)*q_hat[(s,a)] + alpha_t*\
                    (r + self.domain.discount*np.max(np.array([q_hat[(s_prime,a_prime)] for a_prime in self.domain.actions])))
                q_hat[(s,a)] = buffer
            curr_state = s_prime # update current state to next state
            if protocol == 2:
                alpha_t = 0.8 * alpha_t

    def eps_greedy(self, q_hat, s):
        eps = np.random.rand()
        if eps <= self.epsilon:
            return self.rand_policy()
        else:
            return self.opt_policy(q_hat, s)
        
    def rand_policy(self):
        return self.domain.actions[np.random.choice(len(self.domain.actions))]
    
    def opt_policy(self, q_hat, s):
        return self.domain.actions[np.argmax(np.array([q_hat[s,a] for a in self.domain.actions]))]

    def plot_convergence(self, n_episodes, inf_norm, l2_norm, type="det", protocol=1, path="."):
        """Plot the convergence of the estimated parameters to the true parameters."""
        plt.figure(figsize=(10, 6))
        if type == "det":
            domain_name = "deterministic"
        else:
            domain_name = "stochastic"
        
        plt.plot(range(1, n_episodes + 1), inf_norm, label='$||J_{N}^{\\mu_{\\hat{Q}}} - J_{N}^{\\mu^*}||_\\infty$', marker='o')
        plt.title(f"Convergence of State Values against number of episodes for {domain_name} domain under protocol {protocol}")
        plt.xlabel('Number of Episodes')
        plt.ylabel('Infinite Norm of Difference')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        
        plt.savefig(path+f'/evolution_{domain_name}_{protocol}_inf.png')

        plt.figure(figsize=(10, 6))
        if type == "det":
            domain_name = "deterministic"
        else:
            domain_name = "stochastic"
        
        plt.plot(range(1, n_episodes + 1), l2_norm, label='$||J_{N}^{\\mu_{\\hat{Q}}} - J_{N}^{\\mu^*}||_2$', marker='o')
        plt.title(f"Convergence of State Values against number of episodes for {domain_name} domain under protocol {protocol}")
        plt.xlabel('Number of Episodes')
        plt.ylabel('L2 Norm of Difference')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        
        plt.savefig(path+f'/evolution_{domain_name}_{protocol}_L2.png')
    


if __name__ == "__main__":
    sys.setrecursionlimit(10**5)

    n, m = 5, 5  # Grid size
    g = np.array([[-3,   1,  -5, 0,  19],
                  [ 6,   3,   8, 9,  10],
                  [ 5,  -8,   4, 1,  -8],
                  [ 6,  -9,   4, 19, -5],
                  [-20, -17, -4, -3,  9],])

    s0 = (3, 0)  # Initial state

    # Initialize the domain
    domain = Domain(n, m, g, s0, random_state=42)

    print("Section 5.1")
    
    # Initialize our class
    alpha_k = 0.05
    offlineQ = OfflineQ(domain, alpha_k)

    # Deterministic domain
    # Initializing q_hat
    det_q_hat_n = {}
    for i, j, a in product(range(domain.n), range(domain.m), domain.actions):
        s = (i,j)
        det_q_hat_n[(s,a)] = 0

    max_t = 100
    traj_size = 50000
    t_det = max_t
    det_array = []
    for iter in range(10):
        for i, j, a in product(range(domain.n), range(domain.m), domain.actions):
            s = (i,j)
            det_q_hat_n[(s,a)] = 0
        print("Run ", iter+1, " /10 in deterministic domain")
        det_trajectory = offlineQ.generate_sequence(traj_size, s0, type = "det") # deterministic domain
        q_diff_det = []
        t_det = max_t
        for i in range(max_t):
            old_q_hat = det_q_hat_n.copy()
            for transition in det_trajectory:
                offlineQ.update_q_hat(det_q_hat_n, transition)
            c_inf = offlineQ.compute_inf_norm(det_q_hat_n, old_q_hat)
            q_diff_det.append(c_inf)
            if c_inf <= 0.1:
                print("Converged at ", i*traj_size, " sequence length")
                t_det = traj_size * i
                break
            starting_state = det_trajectory[len(det_trajectory)-1][3]
            det_trajectory = offlineQ.generate_sequence(traj_size, starting_state, type = "det") # deterministic domain
        det_array.append(q_diff_det)

    print("Empirical Optimal horizon in deterministic domain: ", t_det)

    
    print("Results deterministic domain")
    for i, j, a in product(range(domain.n), range(domain.m), domain.actions):
        s = (i,j)
        if det_q_hat_n[(s,a)] != 0:
            print(s, a, det_q_hat_n[(s, a)])

    # action to string dictionary
    action_to_str = {(0, 1): 'right', (0, -1): 'left', (1, 0): 'down', (-1, 0): 'up'}

    det_mu_N = offlineQ.opt_policy(det_q_hat_n)
    print("Optimal policy deterministic domain")
    for i, j in product(range(domain.n), range(domain.m)):
        s = (i,j)
        print(s, " Action: ", action_to_str[det_mu_N[s]])

    # Stochastic domain
    # Initializing q_hat
    sto_q_hat_n = {}

    sto_array = []
    t_sto = max_t
    for iter in range(10):
        for i, j, a in product(range(domain.n), range(domain.m), domain.actions):
            s = (i,j)
            sto_q_hat_n[(s,a)] = 0
        print("Run ", iter+1, " /10 in stochastic domain")
        sto_trajectory = offlineQ.generate_sequence(traj_size, s0, type = "sto") # stochastic domain
        q_diff_sto = []
        t_sto = max_t
        for i in range(max_t):
            old_q_hat = sto_q_hat_n.copy()
            for transition in sto_trajectory:
                offlineQ.update_q_hat(sto_q_hat_n, transition)
            c_inf = offlineQ.compute_inf_norm(sto_q_hat_n, old_q_hat)
            q_diff_sto.append(c_inf)
            if c_inf <= 0.1:
                print("Converged at ", i*traj_size, " sequence length")
                t_sto = i * traj_size
                break
            starting_state = sto_trajectory[len(sto_trajectory)-1][3]
            sto_trajectory = offlineQ.generate_sequence(traj_size, starting_state, type = "sto") # stochastic domain
        sto_array.append(q_diff_sto)

    print("Optimal horizon in stocha domain: ", t_sto)

    print("Results stochastic domain")
    for i, j, a in product(range(domain.n), range(domain.m), domain.actions):
        s = (i,j)
        if sto_q_hat_n[(s,a)] != 0:
            print(s, a, sto_q_hat_n[(s, a)])

    sto_mu_N = offlineQ.opt_policy(sto_q_hat_n)
    print("Optimal policy stochastic domain")
    for i, j in product(range(domain.n), range(domain.m)):
        s = (i,j)
        print(s, " Action: ", action_to_str[sto_mu_N[s]])

    det_policy = [det_mu_N[(i,j)] for i, j in product(range(domain.n), range(domain.m))]
    sto_policy = [sto_mu_N[(i,j)] for i, j in product(range(domain.n), range(domain.m))]
    
    # J function for the deterministic domain
    @lru_cache(maxsize=None)
    def det_j_func(domain, state, policy, N):
        if N == 0:
            return 0
        else:
            j = 0
            state_index = state[0] * domain.n + state[1]
            action = policy[state_index]
            j = domain.det_reward(state, action) + domain.discount * det_j_func(domain, domain.dynamic(state, action), policy, N-1)
            return j
        
    # J function for the deterministic domain
    @lru_cache(maxsize=None)
    def sto_j_func(domain, state, policy, N):
        if N == 0:
            return 0
        else:
            j = 0
            state_index = state[0] * domain.n + state[1]
            action = policy[state_index]
            j = 0.5 * (domain.det_reward(state, action) + domain.discount * sto_j_func(domain, domain.dynamic(state, action), policy, N-1))
            j += 0.5 * (domain.g[0, 0] + domain.discount * sto_j_func(domain, (0, 0), policy, N-1))
            return j

    # Value of N we use to estimate J
    N = 10**4

    # Estimate J for our random policy in deterministic domain:
    print('Deterministic domain')
    print("s; J_(mu*)(s)")
    for i in range(domain.n):
        for j in range(domain.m):
            s = (i,j)
            print(f"({i},{j}); {det_j_func(domain, s, tuple(det_policy), N)}")

    # Estimate J for our random policy in stochastic domain:
    print('Stochastic domain')
    print("s; J_(mu*)(s)")
    for i in range(domain.n):
        for j in range(domain.m):
            s = (i,j)
            print(f"({i},{j}); {sto_j_func(domain, s, tuple(sto_policy), N)}")
    
    
    offlineQ.print_policy(det_mu_N, title='Deterministic policy')
    offlineQ.print_policy(sto_mu_N, title='Stochastic policy')
    
    max_length = max(len(lst) for lst in det_array)
    for lst in det_array:
        last_value = lst[-1]
        lst.extend([last_value] * (max_length - len(lst)))

    # Convert the list of lists into a 2D numpy array for easier calculations
    data = np.array(det_array)

    # Calculate the mean and standard deviation along the vertical axis (for each time step across all runs)
    mean_data = np.mean(data, axis=0)
    std_dev_data = np.std(data, axis=0)

    # Create the time steps for plotting
    time_steps = np.arange(max_length)

    # Plotting the mean with the standard deviation as a hull
    plt.figure(figsize=(10, 10))
    plt.plot(time_steps, mean_data, label='Mean')
    plt.fill_between(time_steps, mean_data - std_dev_data, mean_data + std_dev_data, alpha=0.2, label='Std Dev')
    plt.title("Mean and Standard Deviation of Q-value Differences Over Time (Deterministic)")
    plt.xlabel("Time Step")
    plt.ylabel("Q-value Difference in Infinity Norm")
    plt.ylim(0, None)
    plt.grid()
    plt.legend()
    plt.show()

    max_length = max(len(lst) for lst in sto_array)
    for lst in sto_array:
        last_value = lst[-1]
        lst.extend([last_value] * (max_length - len(lst)))

    # Convert the list of lists into a 2D numpy array for easier calculations
    data = np.array(sto_array)

    # Calculate the mean and standard deviation along the vertical axis (for each time step across all runs)
    mean_data = np.mean(data, axis=0)
    std_dev_data = np.std(data, axis=0)

    # Create the time steps for plotting
    time_steps = np.arange(max_length)

    # Plotting the mean with the standard deviation as a hull
    plt.figure(figsize=(10, 10))
    plt.plot(time_steps, mean_data, label='Mean')
    plt.fill_between(time_steps, mean_data - std_dev_data, mean_data + std_dev_data, alpha=0.2, label='Std Dev')

    plt.title("Mean and Standard Deviation of Q-value Differences Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Q-value Difference in Infinity Norm")
    plt.ylim(0, None)
    plt.grid()
    plt.legend()
    plt.show()


    print("Section 5.2")
    
    # Initialize MDP
    domain.set_discount(0.99)
    mdp = MDP(domain)
    
    # Initialize our class
    alpha = 0.05
    epsilon = 0.5
    onlineQ = OnlineQ(domain, alpha, epsilon, mdp)

    # Derive the optimal strategies
    onlineQ.mu_star(type = "det")
    onlineQ.mu_star(type = "sto")
    print("Mu star derived")

    # Derive J_mu_star
    onlineQ.j_star(type = "det")
    onlineQ.j_star(type = "sto")
    print("J star derived")

    # Deriving infinity norm differences
    nbr_transi = 1000
    nbr_ep = 100

    all_diffs = np.empty((6, nbr_ep))

    for i in range(1, 4):
        print(f"Protocol {i}")
        print("Deterministic")
        det_diffs, det_diffs_l2 = onlineQ.sim_episodes(s0, nbr_transi, nbr_ep, type = "det", protocol = i)
        onlineQ.plot_convergence(nbr_ep, det_diffs, det_diffs_l2, type = "det", protocol = i)
        print("Stochastic")
        sto_diffs, det_diffs_l2 = onlineQ.sim_episodes(s0, nbr_transi, nbr_ep, type = "sto", protocol = i)
        onlineQ.plot_convergence(nbr_ep, sto_diffs, det_diffs_l2, type = "sto", protocol = i)

    print("Section 5.3")
 
    # Initialize MDP
    # Set the discount factor to 0.4 as asked in section 5.3
    domain.set_discount(0.4)
    mdp = MDP(domain)

    # Initialize our class
    alpha = 0.05
    epsilon = 0.5
    onlineQ = OnlineQ(domain, alpha, epsilon, mdp)

    # Derive the optimal strategies
    onlineQ.mu_star(type = "det")
    onlineQ.mu_star(type = "sto")
    print("Mu star derived")

    # Derive J_mu_star
    onlineQ.j_star(type = "det")
    onlineQ.j_star(type = "sto")
    print("J star derived")

    # Deriving infinity norm differences
    nbr_transi = 1000
    nbr_ep = 100

    all_diffs = np.empty((6, nbr_ep))

    for i in range(1, 4):
        print(f"Protocol {i}")
        print("Deterministic")
        det_diffs, det_diffs_l2 = onlineQ.sim_episodes(s0, nbr_transi, nbr_ep, type = "det", protocol = i)
        onlineQ.plot_convergence(nbr_ep, det_diffs, det_diffs_l2, type = "det", protocol = i, path="./5.3")
        print("Stochastic")
        sto_diffs, det_diffs_l2 = onlineQ.sim_episodes(s0, nbr_transi, nbr_ep, type = "sto", protocol = i)
        onlineQ.plot_convergence(nbr_ep, sto_diffs, det_diffs_l2, type = "sto", protocol = i, path="./5.3")
