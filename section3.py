from section1 import Domain
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from functools import lru_cache
import sys

class MDP:
    def __init__(self, domain):
        self.domain = domain # Domain to which the MDP is equivalent
        # State and actions defined in the MDP
        self.allowed_sa = [(s, a) for s in product(range(self.domain.n), range(self.domain.m)) for a in self.domain.actions]

    def det_proba(self, s_prime, s, a):
        """
        Derive transition probability in the deterministic domain.
        """
        if self.domain.dynamic(s, a) == s_prime:
            return 1
        else:
            return 0
        
    def sto_proba(self, s_prime, s, a):
        """
        Derive transition probability in the stochastic domain.
        """
        proba = 0.5 * self.det_proba(s_prime, s, a)
        if s_prime == (0,0):
            proba += 0.5
        return proba
    
    def det_rew(self, s, a):
        """
        Derive reward for a given state-action pair in the deterministic domain.
        """
        return self.domain.g[self.domain.dynamic(s, a)]
    
    def sto_rew(self, s, a):
        """
        Derive reward for a given state-action pair in the stochastic domain.
        """
        return (0.5 * self.domain.g[0, 0]) + 0.5 * self.det_rew(s,a)
    
    def get_true_r(self, type='det'):
        """
        Get all rewards of one of the domains.
        """
        r = {}
        if type == 'det':
            for (s, a) in self.allowed_sa:
                r[(s, a)] = self.det_rew(s, a)   
        else:
            for (s, a) in self.allowed_sa:
                r[(s, a)] = self.sto_rew(s, a)
        return r
    
    def get_true_p(self, type='det'):
        """
        Get all transition probabilities of one of the domains.
        """
        p = {}
        if type == 'det':
            for (s, a) in self.allowed_sa:
                p[(s, a)] = {}
                for s_prime in product(range(self.domain.n), range(self.domain.m)):
                    p[(s, a)][s_prime] = self.det_proba(s_prime, s, a)
        else:
            for (s, a) in self.allowed_sa:
                p[(s, a)] = {}
                for s_prime in product(range(self.domain.n), range(self.domain.m)):
                    p[(s, a)][s_prime] = self.sto_proba(s_prime, s, a)
        return p
    
    @lru_cache(maxsize=None) # Use the cache to avoid recomputing several times the same values
    def det_Q_N(self, s, a, N):
        """
        Compute the Q_N-value for a given state-action pair, for the deterministic domain.
        """
        if N == 0:
            return 0
        else:
            cumul = 0
            for i, j in product(range(self.domain.n), range(self.domain.m)):
                s_prime = (i, j)
                cumul += self.det_proba(s_prime, s, a) * np.max(np.array([self.det_Q_N(s_prime, a_prime, N-1) for a_prime in self.domain.actions]))
                
            Q_n = self.det_rew(s, a) + self.domain.discount * cumul
            return Q_n
    
    @lru_cache(maxsize=None) # Use the cache to avoid recomputing several times the same values
    def det_policy(self, N):
        """
        Derive the optimal policy in the deterministic domain.
        """
        mu_N = [np.argmax(np.array([self.det_Q_N(s, a, N) for a in self.domain.actions])) for s in product(range(self.domain.n), range(self.domain.m))]
        return mu_N
    
    @lru_cache(maxsize=None) # Use the cache to avoid recomputing several times the same values
    def sto_Q_N(self, s, a, N):
        """
        Compute the Q_N-value for a given state-action pair, for the stochastic domain.
        """
        if N == 0:
            return 0
        else:
            cumul = 0
            for i, j in product(range(self.domain.n), range(self.domain.m)): 
                s_prime = (i, j)
                cumul += self.sto_proba(s_prime, s, a) * np.max(np.array([self.sto_Q_N(s_prime, a_prime, N-1) for a_prime in self.domain.actions]))
                
            Q_n = self.sto_rew(s, a) + self.domain.discount * cumul
            return Q_n
        
    @lru_cache(maxsize=None) # Use the cache to avoid recomputing several times the same values
    def sto_policy(self, N):
        """
        Derive the optimal policy in the stochastic domain.
        """
        mu_N = [np.argmax(np.array([self.sto_Q_N(s, a, N) for a in self.domain.actions])) for s in product(range(self.domain.n), range(self.domain.m))]
        return mu_N

    def print_policy(self, policy, title='Policy'):
        """
        Visualize policy with arrows on a grid.
        """
        X, Y = np.meshgrid(np.arange(0.5, 5, 1), np.arange(0.5, 5, 1))
        U = np.zeros_like(X, dtype=float)
        V = np.zeros_like(Y, dtype=float)

        # Setting up the arrows based on the policy
        for i in range(5):
            for j in range(5):
                action = policy[i * 5 + j]
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
    # Increase the recursion limit
    sys.setrecursionlimit(10**5)

    n, m = 5, 5  # Grid size
    # Rewards
    g = np.array([[-3,   1,  -5, 0,  19],
                  [ 6,   3,   8, 9,  10],
                  [ 5,  -8,   4, 1,  -8],
                  [ 6,  -9,   4, 19, -5],
                  [-20, -17, -4, -3,  9],])

    s0 = (3, 0)  # Initial state

    # Initialize the domain
    domain = Domain(n, m, g, s0, random_state=42)

    # Initialize MDP
    mdp = MDP(domain)

    # Finding the optimal policy in the deterministic setting
    det_current_policy = mdp.det_policy(0)
    det_n_min = 0
    for n in range(1000):
        det_old_policy = det_current_policy
        det_current_policy = mdp.det_policy(n)

        if det_current_policy != det_old_policy:
            det_n_min = n

    # Action to string dictionary for prettier printing
    action_to_str = {(0, 1): 'right', (0, -1): 'left', (1, 0): 'down', (-1, 0): 'up'}

    # Printing the obtained policy
    det_policy = [domain.actions[i] for i in det_current_policy]
    print("Deterministic policy; lowest N: ", det_n_min)
    for i, j in product(range(domain.n), range(domain.m)):
        s = (i,j)
        print("state ", s, "action: ", action_to_str[det_policy[i * domain.n + j]])

    # Finding the optimal policy in the stochastic setting
    sto_current_policy = mdp.sto_policy(0)
    sto_n_min = 0
    for n in range(1000):
        sto_old_policy = sto_current_policy
        sto_current_policy = mdp.sto_policy(n)

        if sto_current_policy != sto_old_policy:
            sto_n_min = n

    # Printing the obtained policy
    sto_policy = [domain.actions[i] for i in sto_current_policy]
    print("Stochastic policy; lowest N: ", sto_n_min + 1)
    for i, j in product(range(domain.n), range(domain.m)):
        s = (i,j)
        print("state ", s, "action: ", action_to_str[sto_policy[i * domain.n + j]])

    @lru_cache(maxsize=None) # Use the cache to avoid recomputing several times the same values
    def det_j_func(domain, state, policy, N):
        """
        Calculates the value function J_N for a deterministic domain for a given policy
        """
        if N == 0:
            return 0
        else:
            j = 0
            state_index = state[0] * domain.n + state[1]
            action = policy[state_index]
            j = domain.det_reward(state, action) + domain.discount * det_j_func(domain, domain.dynamic(state, action), policy, N-1)
            return j
        
    @lru_cache(maxsize=None) # Use the cache to avoid recomputing several times the same values
    def sto_j_func(domain, state, policy, N):
        """
        Calculates the value function J_N for a stochastic domain for a given policy
        """
        if N == 0:
            return 0
        else:
            j = 0
            state_index = state[0] * domain.n + state[1]
            action = policy[state_index]
            j = 0.5 * (domain.det_reward(state, action) + domain.discount * sto_j_func(domain, domain.dynamic(state, action), policy, N-1))
            j += 0.5 * (domain.g[0, 0] + domain.discount * sto_j_func(domain, (0, 0), policy, N-1))
            return j

    # Value of N used to estimate J
    N = 10**4

    # Estimate J for the optimal policy in deterministic domain:
    print('Deterministic domain')
    print("s; J_(mu*)(s)")
    for i in range(domain.n):
        for j in range(domain.m):
            s = (i,j)
            print(f"({i},{j}); {det_j_func(domain, s, tuple(det_policy), N)}")

    # Estimate J for the optimal policy in stochastic domain:
    print('Stochastic domain')
    print("s; J_(mu*)(s)")
    for i in range(domain.n):
        for j in range(domain.m):
            s = (i,j)
            print(f"({i},{j}); {sto_j_func(domain, s, tuple(sto_policy), N)}")

    # Printing the figures representing the optimal policies
    mdp.print_policy(det_policy, title='Deterministic policy')
    mdp.print_policy(sto_policy, title='Stochastic policy')
