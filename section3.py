from section1 import Domain
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from functools import lru_cache
import sys

class MDP:
    def __init__(self, domain):
        self.domain = domain

    def det_proba(self, s_prime, s, a):
        if self.domain.dynamic(s, a) == s_prime:
            return 1
        else:
            return 0
        
    def sto_proba(self, s_prime, s, a):
        proba = 0.5 * self.det_proba(s_prime, s, a)
        if s_prime == (0,0):
            proba += 0.5
        return proba
    
    def det_rew(self, s, a):
        return self.domain.g[self.domain.dynamic(s, a)]
    
    def sto_rew(self, s, a):
        return (0.5 * self.domain.g[0, 0]) + 0.5 * self.det_rew(s,a)
    
    @lru_cache(maxsize=None)
    def det_Q_N(self, s, a, N):
        if N == 0:
            return 0
        else:
            cumul = 0
            for i, j in product(range(self.domain.n), range(self.domain.m)):
                s_prime = (i, j)
                cumul += self.det_proba(s_prime, s, a) * np.max(np.array([self.det_Q_N(s_prime, a_prime, N-1) for a_prime in self.domain.actions]))
                
            Q_n = self.det_rew(s, a) + self.domain.discount * cumul
            return Q_n
    
    @lru_cache(maxsize=None)
    def det_policy(self, N):
        mu_N = [np.argmax(np.array([self.det_Q_N(s, a, N) for a in self.domain.actions])) for s in product(range(self.domain.n), range(self.domain.m))]
        return mu_N
    
    @lru_cache(maxsize=None)
    def sto_Q_N(self, s, a, N):
        if N == 0:
            return 0
        else:
            cumul = 0
            for i, j in product(range(self.domain.n), range(self.domain.m)):
                s_prime = (i, j)
                cumul += self.sto_proba(s_prime, s, a) * np.max(np.array([self.sto_Q_N(s_prime, a_prime, N-1) for a_prime in self.domain.actions]))
                
            Q_n = self.sto_rew(s, a) + self.domain.discount * cumul
            return Q_n
        
    @lru_cache(maxsize=None)
    def sto_policy(self, N):
        mu_N = [np.argmax(np.array([self.sto_Q_N(s, a, N) for a in self.domain.actions])) for s in product(range(self.domain.n), range(self.domain.m))]
        return mu_N



    def print_policy(self, policy, title='Policy'):
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
    sys.setrecursionlimit(10**5)

    n, m = 5, 5  # Grid size
    g = np.array([[-3,   1,  -5, 0,  19],
                  [ 6,   3,   8, 9,  10],
                  [ 5,  -8,   4, 1,  -8],
                  [ 6,  -9,   4, 19, -5],
                  [-20, -17, -4, -3,  9],])

    print(g[0, 0], g[0,1])
    s0 = (3, 0)  # Initial state

    # Initialize the domain
    domain = Domain(n, m, g, s0, random_state=42)

    # Initialize MDP
    mdp = MDP(domain)

    # Finding the deterministic policy such that it doesn't change from one iteration to another
    n = 0
    det_current_policy = mdp.det_policy(0)
    while True:
        n += 1
        det_old_policy = det_current_policy
        det_current_policy = mdp.det_policy(n)

        if det_current_policy == det_old_policy:
            break


    # action to string dictionary
    action_to_str = {(0, 1): 'right', (0, -1): 'left', (1, 0): 'down', (-1, 0): 'up'}

    # print(n, det_current_policy)
    det_policy = [domain.actions[i] for i in det_current_policy]
    print("Deterministic policy")
    for i, j in product(range(domain.n), range(domain.m)):
        s = (i,j)
        print("state ", s, "action: ", action_to_str[det_policy[i * domain.n + j]])

    # Finding the stochastic policy such that it doesn't change from one iteration to another
    n = 0
    sto_current_policy = mdp.sto_policy(0)
    while True:
        n += 1
        sto_old_policy = sto_current_policy
        sto_current_policy = mdp.sto_policy(n)

        if sto_current_policy == sto_old_policy:
            break

    # print(n, sto_current_policy)
    sto_policy = [domain.actions[i] for i in sto_current_policy]
    print("Stochastic policy")
    for i, j in product(range(domain.n), range(domain.m)):
        s = (i,j)
        print("state ", s, "action: ", action_to_str[sto_policy[i * domain.n + j]])
 
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
    for i in range(n):
        for j in range(m):
            s = (i,j)
            print(f"({i},{j}); {det_j_func(domain, s, tuple(det_policy), N)}")

    # Estimate J for our random policy in stochastic domain:
    print('Stochastic domain')
    print("s; J_(mu*)(s)")
    for i in range(n):
        for j in range(m):
            s = (i,j)
            print(f"({i},{j}); {sto_j_func(domain, s, tuple(sto_policy), N)}")

    mdp.print_policy(det_policy, title='Deterministic policy')
    mdp.print_policy(sto_policy, title='Stochastic policy')