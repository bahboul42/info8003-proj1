from section1 import Domain
from itertools import product
import numpy as np
import sys
from functools import lru_cache
from section3 import MDP
import matplotlib.pyplot as plt

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
            opt_policy = self.mdp.det_policy(n)
        else:
            self.mdp.det_policy.cache_clear() # Clearing the cache
            self.mdp.det_Q_N.cache_clear() # Clearing the cache
            opt_policy = self.mdp.sto_policy(n)

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
            else:
                differences[ep] = np.max(np.abs(j_mu_hat - self.sto_J_mu_star))

        return differences

    
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
                    q_hat[(s_up,a_up)] = (1-alpha_t)*q_hat[(s_up,a_up)] + alpha_t*(r_up + self.domain.discount*np.max(np.array([q_hat[(s_prime_up,a_prime)] for a_prime in self.domain.actions])))
            else:
                q_hat[(s,a)] = (1-alpha_t)*q_hat[(s,a)] + alpha_t*(r + self.domain.discount*np.max(np.array([q_hat[(s_prime,a_prime)] for a_prime in self.domain.actions])))
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
    
def plot_convergence(n_episodes, inf_norm, type="det", protocol=1):
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
    
    plt.savefig(f'evolution_{domain_name}_{protocol}.png')


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
        det_diffs = onlineQ.sim_episodes(s0, nbr_transi, nbr_ep, type = "det", protocol = i)
        plot_convergence(nbr_ep, det_diffs, type = "det", protocol = i)
        print("Stochastic")
        sto_diffs = onlineQ.sim_episodes(s0, nbr_transi, nbr_ep, type = "sto", protocol = i)
        plot_convergence(nbr_ep, sto_diffs, type = "sto", protocol = i)
