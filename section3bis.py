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
        
    def det_Q_N(self, q_n1):
        q_n = np.zeros_like(q_n1)
        for i, j in product(range(q_n.shape[0]), range(q_n.shape[1])):
            s = (i // self.domain.n, i % self.domain.n)
            a = self.domain.actions[j]

            rew = self.det_rew(s, a)

            cumul = 0
            for k, l in product(range(self.domain.n), range(self.domain.m)):
                s_prime = (k, l)
                s_prime_index = k * self.domain.n + l
                cumul += self.det_proba(s_prime, s, a) * np.max(np.array([q_n1[s_prime_index, a_prime] for a_prime in range(len(self.domain.actions))]))
            
            q_n[i, j] = rew + self.domain.discount * cumul
        return q_n
    
    def sto_Q_N(self, q_n1):
        q_n = np.zeros_like(q_n1)
        for i, j in product(range(q_n.shape[0]), range(q_n.shape[1])):
            s = (i // self.domain.n, i % self.domain.n)
            a = self.domain.actions[j]

            rew = self.sto_rew(s, a)

            cumul = 0
            for k, l in product(range(self.domain.n), range(self.domain.m)):
                s_prime = (k, l)
                s_prime_index = k * self.domain.n + l
                cumul += self.sto_proba(s_prime, s, a) * np.max(np.array([q_n1[s_prime_index, a_prime] for a_prime in range(len(self.domain.actions))]))
            
            q_n[i, j] = rew + self.domain.discount * cumul
        return q_n
                
    def opt_policy(self, q_n):
        return np.argmax(q_n, axis = 1)

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

    s0 = (3, 0)  # Initial state

    # Initialize the domain
    domain = Domain(n, m, g, s0, random_state=42)

    # Initialize MDP
    mdp = MDP(domain)

    # Finding the optimal policy in the deterministic case
    det_q_0 = np.zeros((domain.g.size, len(domain.actions)))
    det_q_n = det_q_0
    det_current_policy = np.empty(len(domain.actions))

    det_n_min = 0

    for n in range(1000):   
        q_old = det_q_n
        det_q_n = mdp.det_Q_N(q_old)

        old_policy = det_current_policy
        det_current_policy = mdp.opt_policy(det_q_n)

        if not np.array_equal(old_policy, det_current_policy):
            det_n_min = n
        if np.array_equal(q_old, det_q_n):
            print('Q has converged after ', n+1, ' iterations')
            break
    
    # action to string dictionary
    action_to_str = {(0, 1): 'right', (0, -1): 'left', (1, 0): 'down', (-1, 0): 'up'}

    # print(n, det_current_policy)
    det_policy = [domain.actions[i] for i in det_current_policy]
    print("Deterministic policy; lowest N: ", det_n_min + 1)
    for i, j in product(range(domain.n), range(domain.m)):
        s = (i,j)
        print("state ", s, "action: ", action_to_str[det_policy[i * domain.n + j]])

    # Finding the optimal policy in the stochastic case
    sto_q_0 = np.zeros((domain.g.size, len(domain.actions)))
    sto_q_n = sto_q_0
    sto_current_policy = np.empty(len(domain.actions))

    sto_n_min = 0

    for n in range(1000):   
        q_old = sto_q_n
        sto_q_n = mdp.sto_Q_N(q_old)

        old_policy = sto_current_policy
        sto_current_policy = mdp.opt_policy(sto_q_n)

        if not np.array_equal(old_policy, sto_current_policy):
            sto_n_min = n
        if np.array_equal(q_old, sto_q_n):
            print('Q has converged after ', n+1, ' iterations')
            break

    sto_policy = [domain.actions[i] for i in sto_current_policy]
    print("Stochastic policy; lowest N: ", sto_n_min)
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
            print(state[0], state[1], state_index)
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

    mdp.print_policy(det_policy, title='Deterministic policy')
    mdp.print_policy(sto_policy, title='Stochastic policy')