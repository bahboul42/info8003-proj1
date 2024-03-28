import numpy as np
import matplotlib.pyplot as plt
from section1 import Domain
from section1 import *

class PolicyEstimator:
    def __init__(self, domain, agent):
        self.domain = domain
        self.agent = agent

    def policy_return(self, N, n_initials):
        '''Derive the estimated expected returns for horizon up to N, using n_initials starting states'''
        est_return = np.zeros((n_initials, N))

        for i in range(n_initials):
            self.domain.reset() # Reset the domain
            cum_reward = 0

            for j in range(N):
                state = self.domain.get_state() # Current state
                action = self.agent.get_action(state) # Action chosen by policy for current state
                _, _, r, _ = self.domain.step(action) # Move system one step forward and store the reward
                cum_reward += (self.domain.discount ** j) * r # Update cumulative reward
                est_return[i, j] = cum_reward # Store current cumulative reward

        return est_return
    
    def plot_return(self, all_returns, filename = "", path="../../figures/project2/section2"):
        """Plot the evolution of the estimated expected return against horizon N."""
        
        n_initials = all_returns.shape[0]
        N = all_returns.shape[1]

        plt.figure(figsize=(10, 6))
        for i in range(n_initials):
            plt.plot(range(1, N + 1), all_returns[i, :], color = 'blue')
        plt.plot(range(1, N+1), np.mean(all_returns, axis = 0), color = 'red', label = f'Average over {n_initials} trajectories')
        plt.title(f"Convergence of expected return against N")
        plt.xlabel('N')
        plt.ylabel('Expected return')
        plt.xlim((1, N+1))
        plt.legend()
        plt.grid(True, which="both", ls="--")
        
        plt.savefig(path+f'/conv_exp_return{filename}.png')
        plt.close()


if __name__ == "__main__":
    np.random.seed(0)

    domain = Domain() # Create the environment
    domain.sample_initial_state() # Sample an initial state

    agent = MomentumAgent() # Create the agent
    # agent = AcceleratingAgent()
    # agent = RandomAgent()
    policy_est = PolicyEstimator(domain, agent)

    n_initials = 50 # Number of initial states
    N = 300 # Horizon

    all_returns = policy_est.policy_return(N, n_initials) # Get all the estimated expected returns

    policy_est.plot_return(all_returns) # Plot the returns for the 50 initial states

    # Inspecting the final expected return of all trajectories to know in more detail
    count_pos = 0
    count_neg = 0
    for i in range(n_initials):
        if all_returns[i, -1] > 0:
            count_pos += 1
        elif all_returns[i, -1] < 0:
            count_neg += 1
    print(f'Number of successful trajectories: {count_pos}')
    print(f'Number of unsuccessful trajectories: {count_neg}')
