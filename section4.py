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
        self.r_hat = {} # Estimated reward dictionary
        self.p_hat = {} # Estimated transition probability dictionary
        self.allowed_sa = [(s, a) for s in product(range(self.domain.n), range(self.domain.m)) for a in self.domain.actions] # Allowed state-action pairs
        self.compute_r_hat(initial_sequence)
        self.compute_p_hat(initial_sequence)        

    
    def compute_r_hat(self, sequence):
        """Compute the estimated reward for each state-action pair."""
        reward_sums = {}
        counts = {}
        
        # iterate through the sequence and update the reward sums and counts
        for s, a, r in sequence:
            key = (s, a) 
            
            if key not in self.allowed_sa:
                raise ValueError(f"Invalid state-action pair: {key}")
            
            # Update the reward sum and count for each s-a pair
            if key in reward_sums:
                reward_sums[key] += r
                counts[key] += 1
            else:
                reward_sums[key] = r
                counts[key] = 1
        
        # Calculate the average reward for each s-a pair
        self.r_hat = {key: reward_sums[key] / counts[key] for key in reward_sums}
        
    def get_rew(self, s, a):
        # Extract the reward from the dict for a given state-action pair
        if (s, a) not in self.r_hat:
            return 0
        return self.r_hat.get((s, a), 0)


    def compute_p_hat(self, sequence):
        """Compute the estimated transition probabilities for each state-action pair."""
        trans = []
        for i in range(len(sequence)-1):  # do not explore last state to not segfault
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

        # Iterate through the sequence and update the transitions and counts
        for s, a, s_prime in trans:
            if (s, a) not in transitions:
                transitions[(s, a)] = {}
                counts[(s, a)] = 0
            if s_prime not in transitions[(s, a)]:
                transitions[(s, a)][s_prime] = 0
            transitions[(s, a)][s_prime] += 1
            counts[(s, a)] += 1

        p_hat = {}
        # Calculate the transition probabilities for each state-action pair
        for state_action, next_states in transitions.items():
            p_hat[state_action] = {}
            for s_prime, count in next_states.items():
                p_hat[state_action][s_prime] = count / counts[state_action]

        self.p_hat = p_hat

    def get_proba(self, s_prime, s, a):
        """Return the estimated probability of transitioning to state s_prime from state s under action a."""
        return self.p_hat.get((s, a), {}).get(s_prime, 0) # Return 0 if the probability is not found


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
    
    def _rand_policy(self): return domain.actions[np.random.choice(len(domain.actions))] # Random policy as asked in the assignment

    def get_sequence(self, sequence_len, s0, type="det"):
        """Generate a sequence of states, actions, and rewards for Deterministic domain"""
        sequence = []

        curr_state = s0
        # Generate a sequence of states, actions, and rewards for the given sequence length
        for _ in range(sequence_len):
            state = curr_state
            action = self._rand_policy() # get random action
            if type == "det":
                next_state = domain.det_dyn(state, action) # get next state from action and current state
                reward = domain.det_reward(state, action) # get reward from action and current state
            else:
                next_state = domain.sto_dyn(state, action) # get next state from action and current state
                reward = domain.sto_reward(state, action) # get reward from action and current state
            sequence.append((state, action, reward)) # append to sequence
            curr_state = next_state # update current state to next state
        return sequence

    def compute_inf_norm(self, pred, true, type="r"):
        """Compute the infinity norm of the difference between the true and estimated parameters."""
        max_diff = 0

        # Fetch the true and estimated parameters and compute the difference, only keep the maximum difference
        if type == "r" or type == "q":
            for key in self.allowed_sa:
                true_r = true.get(key, 0)
                estimated_r = pred.get(key, 0)
                diff = np.abs(true_r - estimated_r)
                if diff > max_diff:
                    max_diff = diff
        elif type == "p":
            for key in self.allowed_sa:
                for s_prime in product(range(self.domain.n), range(self.domain.m)):
                    true_p = (true.get(key, {})).get(s_prime, 0)
                    estimated_p = (pred.get(key, {})).get(s_prime, 0)
                    [s_prime]
                    diff = np.abs(true_p - estimated_p)
                    if diff > max_diff:
                        max_diff = diff
        else:
            raise ValueError(f"Invalid type: {type}")
        
        return max_diff

def plot_convergence(trajectory_lengths, inf_norm, type="r", fname='foobar'):
    """Plot the convergence of the estimated parameters to the true parameters."""
    plt.figure(figsize=(10, 6))
    
    if type == "r":
        plt.plot(trajectory_lengths, inf_norm, label='Reward Convergence ($||\\hat{r} - r||_\\infty$)', marker='o')
    elif type == "p":
        plt.plot(trajectory_lengths, inf_norm, label='Probability Convergence ($||\\hat{p} - p||_\\infty$)', marker='x')
    elif type == "q":
        plt.plot(trajectory_lengths, inf_norm, label='Q-Value Convergence ($||\\hat{q} - q||_\\infty$)', marker='s')
    else:
        raise ValueError(f"Invalid type: {type}")
    
    plt.title('Convergence of Estimated Parameters to True Values')
    plt.xlabel('Trajectory Length')
    plt.ylabel('Infinite Norm of Difference')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    plt.savefig(f'convergence_{fname}.png')

# J function for the deterministic domain
@lru_cache(maxsize=None)
def det_j_func(domain, state, policy, N, mdp):
    if N == 0:
        return 0
    else:
        j = 0
        state_index = state[0] * domain.n + state[1]
        action = policy[state_index]
        j = mdp.get_rew(state, action) + domain.discount * det_j_func(domain, domain.dynamic(state, action), policy, N-1, mdp)
        return j
    
# J function for the deterministic domain
@lru_cache(maxsize=None)
def sto_j_func(domain, state, policy, N, mdp):
    if N == 0:
        return 0
    else:
        j = 0
        state_index = state[0] * domain.n + state[1]
        action = policy[state_index]
        j = 0.5 * (mdp.get_rew(state, action) + domain.discount * sto_j_func(domain, domain.dynamic(state, action), policy, N-1, mdp))
        j += 0.5 * (domain.g[0, 0] + domain.discount * sto_j_func(domain, (0, 0), policy, N-1, mdp))
        return j
    

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
    mdp_e_sto = MDPEstimator(domain)
    mdp = MDP(domain)


    trajectory_lengths = np.arange(2, 1300, 40)
    inf_norm_r = []
    inf_norm_p = []
    inf_norm_q = []

    inf_norm_r_sto = []
    inf_norm_p_sto = []
    inf_norm_q_sto = []


    # get True R and P
    true_r = mdp.get_true_r()
    true_p = mdp.get_true_p()
    true_q = mdp.get_true_q()

    true_r_sto = mdp.get_true_r(type="sto")
    true_p_sto = mdp.get_true_p(type="sto")
    true_q_sto = mdp.get_true_q(type="sto")


    det_policy = []
    sto_policy = []

    # from multiprocessing import ThreadPool


    # for i, length in enumerate(trajectory_lengths):
    #     print(f'"det" iter {i+1}/{len(trajectory_lengths)}: trajectory length = {length}')

    #     ## DETERMINISTIC DOMAIN
    #     sequence = mdp_e.get_sequence(length, s0)
        
    #     mdp_e.compute_r_hat(sequence)
    #     mdp_e.compute_p_hat(sequence)
        
    #     inf_norm_r.append(mdp_e.compute_inf_norm(mdp_e.r_hat, true_r, type="r"))
    #     inf_norm_p.append(mdp_e.compute_inf_norm(mdp_e.p_hat, true_p, type="p"))

    #     mdp_e.Q_N.cache_clear()
    #     q_hat = {}
    #     for key in mdp_e.allowed_sa:
    #         s, a = key
    #         test = mdp_e.Q_N(s, a, 600)
    #         q_hat[key] = test
    #     inf_norm_q.append(mdp_e.compute_inf_norm(q_hat, true_q, type="q"))

    #     det_policy_current = [max(domain.actions, key=lambda a: q_hat[(s, a)]) for s in product(range(domain.n), range(domain.m))]
    
    trajectory_lengths = np.arange(12000, 32041, 1000)
    for i, length in enumerate(trajectory_lengths):
        print(f'"sto" iter {i+1}/{len(trajectory_lengths)}: trajectory length = {length}')

        ## STOCHASTIC DOMAIN
        sequence_sto = mdp_e_sto.get_sequence(length, s0, type="sto")
        
        mdp_e_sto.compute_r_hat(sequence_sto)
        mdp_e_sto.compute_p_hat(sequence_sto)
        
        inf_norm_r_sto.append(mdp_e_sto.compute_inf_norm(mdp_e.r_hat, true_r_sto, type="r"))
        inf_norm_p_sto.append(mdp_e_sto.compute_inf_norm(mdp_e.p_hat, true_p_sto, type="p"))

        mdp_e_sto.Q_N.cache_clear()
        q_hat_sto = {}
        for key in mdp_e.allowed_sa:
            s, a = key
            test = mdp_e_sto.Q_N(s, a, 600)
            q_hat_sto[key] = test
        inf_norm_q_sto.append(mdp_e_sto.compute_inf_norm(q_hat_sto, true_q_sto, type="q"))
        sto_policy_current = [max(domain.actions, key=lambda a: q_hat_sto[(s, a)]) for s in product(range(domain.n), range(domain.m))]

    # Visualization of convergence
    # plot_convergence(trajectory_lengths, inf_norm_r, type="r", fname='r_d')
    # plot_convergence(trajectory_lengths, inf_norm_p, type="p", fname='p_d')
    # plot_convergence(trajectory_lengths, inf_norm_q, type="q", fname='q_d')

    plot_convergence(trajectory_lengths, inf_norm_r_sto, type="r", fname='r_sto')
    plot_convergence(trajectory_lengths, inf_norm_p_sto, type="p", fname='p_sto')
    plot_convergence(trajectory_lengths, inf_norm_q_sto, type="q", fname='q_sto')

    # det_policy = det_policy_current
    sto_policy = sto_policy_current

    # Value of N we use to estimate J
    N = 10**4

    # Estimate J for our random policy in deterministic domain:
    # print('Deterministic domain')
    # print("s; J_(mu*)(s)")
    # for i in range(domain.n):
    #     for j in range(domain.m):
    #         s = (i,j)
    #         print(f"({i},{j}); {det_j_func(domain, s, tuple(det_policy), N, mdp_e)}")

    # Estimate J for our random policy in stochastic domain:
    print('Stochastic domain')
    print("s; J_(mu*)(s)")
    for i in range(domain.n):
        for j in range(domain.m):
            s = (i,j)
            print(f"({i},{j}); {sto_j_func(domain, s, tuple(sto_policy), N, mdp_e_sto)}")

    # mdp.print_policy(det_policy, title='Deterministic policy')
    mdp.print_policy(sto_policy, title='Stochastic policy')
