import numpy as np
import matplotlib.pyplot as plt
from section1 import Domain
# from section1 import AcceleratingAgent
from section1 import MomentumAgent

class PolicyEstimator:
    def __init__(self, domain, agent):
        self.domain = domain
        self.agent = agent

    def policy_return(self, N, n_initials):
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
    
    def plot_return(self, N, n_initials, path="../../figures/project2/section2"):
        """Plot the evolution of the estimated expected return against horizon N."""
        evol_return = self.policy_return(N, n_initials)

        plt.figure(figsize=(10, 6))
        for i in range(n_initials):
            plt.plot(range(1, N + 1), evol_return[i, :], color = 'blue')
        plt.plot(range(1, N+1), np.mean(evol_return, axis = 0), color = 'red', label = 'Average over 50 trajectories')
        plt.title(f"Convergence of expected return against N")
        plt.xlabel('N')
        plt.ylabel('Expected return')
        plt.xlim((1, N+1))
        plt.legend()
        plt.grid(True, which="both", ls="--")
        
        plt.savefig(path+f'/conv_exp_return.png')


if __name__ == "__main__":
    domain = Domain() # Create the environment
    domain.sample_initial_state() # Sample an initial state

    agent = MomentumAgent() # Create the agent
    # agent = AcceleratingAgent()
    policy_est = PolicyEstimator(domain, agent)

    n_initials = 50 # Number of initial states
    N = 4000 # Horizon
    policy_est.plot_return(4000, n_initials)




