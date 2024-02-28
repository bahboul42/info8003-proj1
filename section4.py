from section1 import *
from section2 import *
from section3 import *

from section1 import Domain
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from functools import lru_cache
import sys

class MDPEstimator:
    def __init__(self, domain, initial_sequence=[]):
        self.domain = domain
        self.r_hat = {}
        self.p_hat = {}
        self.allowed_sa = [(s, a) for s in product(range(self.domain.n), range(self.domain.m)) for a in self.domain.actions]
        self.compute_r_hat(initial_sequence)
        self.compute_p_hat(initial_sequence)        

    
    def compute_r_hat(self, sequence):
        reward_sums = {}
        counts = {}
        
        for s, a, r in sequence:
            key = (s, a) 
            
            if key not in self.allowed_sa:
                raise ValueError(f"Invalid state-action pair: {key}")
            
            if key in reward_sums:
                reward_sums[key] += r
                counts[key] += 1
            else:
                reward_sums[key] = r
                counts[key] = 1
        
        # Calculate the average reward for each s-a pair
        self.r_hat = {key: reward_sums[key] / counts[key] for key in reward_sums}
    
    def update_r_hat(self, s, a, r):
        key = (s, a)
        if key not in self.allowed_sa:
            raise ValueError(f"Invalid state-action pair: {key}")
        if key in self.r_hat:
            self.r_hat[key] = (self.r_hat[key] + r) / 2
        else:
            self.r_hat[key] = r
        
    def get_rew(self, s, a):
        if (s, a) not in self.r_hat:
            return 0
        return self.r_hat[(s, a)]


    def compute_p_hat(self, sequence):
        trans = []
        for i in range(len(sequence)-1):  # do not expect last state to not segfault
            current_seq = sequence[i]
            next_seq = sequence[i+1]  
            s = current_seq[0]  # Current state
            a = current_seq[1]  # Action
            if (s, a) not in self.allowed_sa:
                raise ValueError(f"Invalid state-action pair: {s}, {a}")
            
            s_prime = next_seq[0]  # Next state from the next sequence's current state

            adjusted_seq = (s, a, s_prime)
            trans.append(adjusted_seq)


        transitions = {}  # For counting transitions
        counts = {}  # For counting occurrences of state-action pairs
        for s, a, s_prime in trans:
            if (s, a) not in transitions:
                transitions[(s, a)] = {}
                counts[(s, a)] = 0
            if s_prime not in transitions[(s, a)]:
                transitions[(s, a)][s_prime] = 0
            transitions[(s, a)][s_prime] += 1
            counts[(s, a)] += 1

        p_hat = {}
        for state_action, next_states in transitions.items():
            p_hat[state_action] = {}
            for s_prime, count in next_states.items():
                p_hat[state_action][s_prime] = count / counts[state_action]

        self.p_hat = p_hat

    def update_p_hat(self, s, a, s_prime):
        if (s, a) not in self.allowed_sa:
            raise ValueError(f"Invalid state-action pair: {s}, {a}")
        if (s, a) not in self.p_hat:
            self.p_hat[(s, a)] = {}
        if s_prime not in self.p_hat[(s, a)]:
            self.p_hat[(s, a)][s_prime] = 0
        self.p_hat[(s, a)][s_prime] = (self.p_hat[(s, a)][s_prime] + 1) / 2


    def get_proba(self, s_prime, s, a):
        if s_prime not in self.p_hat[(s, a)]:
            return 0
        return self.p_hat[(s, a)][s_prime]

    
    @lru_cache(maxsize=None)
    def Q_N(self, s, a, N):
        if N == 0:
            return 0
        else:
            cumul = 0
            for i, j in product(range(self.domain.n), range(self.domain.m)):
                s_prime = (i, j)
                cumul += self.get_proba(s_prime, s, a) * np.max(np.array([self.Q_N(s_prime, a_prime, N-1) for a_prime in self.domain.actions]))
                
            Q_n = self.get_rew(s, a) + self.domain.discount * cumul
            return Q_n
    
    @lru_cache(maxsize=None)
    def get_policy(self, N):
        mu_N = [np.argmax(np.array([self.Q_N(s, a, N) for a in self.domain.actions])) for s in product(range(self.domain.n), range(self.domain.m))]
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

    s0 = (3, 0)  # Initial state

    # Initialize the domain
    domain = Domain(n, m, g, s0, random_state=42)

    # Initialize MDP
    mdp_e = MDPEstimator(domain)  
    mdp = MDP(domain)

    sequence_len = 250000
    def rand_policy():
        return domain.actions[np.random.choice(len(domain.actions))]
    
    # Generate a sequence of states, actions, and rewards for Deterministic domain
    sequence = []

    curr_state = s0
    for _ in range(sequence_len):
        state = curr_state
        action = rand_policy()
        next_state = domain.sto_dyn(state, action)
        reward = domain.sto_reward(state, action)
        sequence.append((state, action, reward))    
        curr_state = next_state
    
    # Estimate the rewards and transition probabilities
    mdp_e.compute_r_hat(sequence)
    mdp_e.compute_p_hat(sequence)

    print("Estimated rewards")
    print(mdp_e.r_hat)

