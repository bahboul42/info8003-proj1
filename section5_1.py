from section1 import Domain
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from functools import lru_cache
import sys

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
        q_hat_n[(s,a)] = (1-self.alpha_k)*q_hat_n[(s,a)] + self.alpha_k*(r + self.domain.discount*np.max(np.array([q_hat_n[(s_prime,a_prime)] for a_prime in self.domain.actions])))

    def opt_policy(self, q_hat):
        mu_N = {}
        for i, j in product(range(self.domain.n), range(self.domain.m)):
            s = (i,j)
            mu_N[s] = self.domain.actions[np.argmax(np.array([q_hat[s,a] for a in self.domain.actions]))]
            # mu_N[s] = np.argmax(np.array([q_hat[s,a] for a in self.domain.actions]))
            
        return mu_N
    
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

    # Initialize our class
    alpha_k = 0.05
    offlineQ = OfflineQ(domain, alpha_k)

    # Generate the trajectory
    det_trajectory = offlineQ.generate_sequence(1000000, s0, type = "det") # deterministic domain
    sto_trajectory = offlineQ.generate_sequence(1000000, s0, type = "sto") # stochastic domain

    # Deterministic domain
    # Initializing q_hat
    det_q_hat_n = {}
    for i, j, a in product(range(domain.n), range(domain.m), domain.actions):
        s = (i,j)
        det_q_hat_n[(s,a)] = 0

    for transition in det_trajectory:
        offlineQ.update_q_hat(det_q_hat_n, transition)
    
    print("Results deterministic domain")
    for i, j, a in product(range(domain.n), range(domain.m), domain.actions):
        s = (i,j)
        if det_q_hat_n[(s,a)] != 0:
            print(s, a, det_q_hat_n[(s, a)])

    # action to string dictionary
    action_to_str = {(0, 1): 'right', (0, -1): 'left', (1, 0): 'down', (-1, 0): 'up'}

    det_mu_N = offlineQ.opt_policy(det_q_hat_n)
    print("Optimal policy deterministic domain")
    for i, j, a in product(range(domain.n), range(domain.m), domain.actions):
        s = (i,j)
        print(s, " Action: ", action_to_str[det_mu_N[s]])

    # Stochastic domain
    # Initializing q_hat
    sto_q_hat_n = {}
    for i, j, a in product(range(domain.n), range(domain.m), domain.actions):
        s = (i,j)
        sto_q_hat_n[(s,a)] = 0

    for transition in sto_trajectory:
        offlineQ.update_q_hat(sto_q_hat_n, transition)
    
    print("Results stochastic domain")
    for i, j, a in product(range(domain.n), range(domain.m), domain.actions):
        s = (i,j)
        if sto_q_hat_n[(s,a)] != 0:
            print(s, a, sto_q_hat_n[(s, a)])
    
    sto_mu_N = offlineQ.opt_policy(det_q_hat_n)
    print("Optimal policy Stochastic domain")
    for i, j, a in product(range(domain.n), range(domain.m), domain.actions):
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







