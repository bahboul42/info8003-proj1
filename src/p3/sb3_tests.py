import gymnasium as gym
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt

# Setting the random seed to obtain reproducible results
seed_value = 42
np.random.seed(seed_value)

# Setting some parameters
n_steps = 500 # How often do we evaluated expected reward
max_steps = 50000 # Maximum number of timesteps

class MeanRewardCallback(BaseCallback):
    '''Callback used to store the expected reward of the current policy after a given number of timesteps.'''
    def __init__(self, env, time_steps=n_steps, verbose=0):
        super(MeanRewardCallback, self).__init__(verbose)
        self.env = env
        self.mean_rewards = []
        self.time_steps = time_steps

    def _on_step(self) -> bool:
        if self.num_timesteps % self.time_steps == 0:
            # Calculate expected reward
            mean_reward = self.evaluate_mean_reward()
            self.mean_rewards.append(mean_reward)
            print(f"Num timesteps: {self.num_timesteps}, Mean reward: {mean_reward}")
        return True

    def evaluate_mean_reward(self) -> float:
        # Compute expected reward of the current policy
        total_reward = 0.0
        n_episodes = 10  # Number of episodes used to evaluate
        vec_env = self.model.get_env()
        for _ in range(n_episodes):
            obs = vec_env.reset()
            episode_reward = 0.0
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = vec_env.step(action)
                episode_reward += reward
                if done:
                    break
            total_reward += episode_reward
        return total_reward / n_episodes
    
def evaluate_model(alg="ddpg",double=False, max_steps=50000, n_runs=3):
    '''Function used to evaluate how expected reward evolves with the number of transitions used
      for a given model.'''
    
    all_rew = [] # Store the evolution of expected reward of the policy after each run

    for i in range(1, n_runs+1):
        print(f'Starting run {i}...')
        # Create environment
        if double:
            env = gym.make("InvertedDoublePendulum-v4")
            # Standard deviation of the noise added to the actions in DDPG
            sigma_std = 0.3
        else:
            env = gym.make("InvertedPendulum-v4")
            sigma_std = 0.5
        
        # Create the model
        if alg == "ddpg":
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=sigma_std * np.ones(n_actions))
            model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=0)
        elif alg == "ppo":
            model = PPO("MlpPolicy", env, verbose=0)
        elif alg == "sac":
            model = SAC("MlpPolicy", env, verbose=0)
        else:
            print("Invalid algorithm.")
            return
        
        # Create callback
        mean_reward_callback = MeanRewardCallback(env)
        
        # Train the model using the callback
        model.learn(total_timesteps=max_steps, callback=mean_reward_callback)
        all_rew.append(mean_reward_callback.mean_rewards)

    return all_rew

def plot_evolution(all_rew, alg="ddpg", double=False):
    '''Function used to plot the evolution of expected reward of the policy.'''
    all_rew = np.array(all_rew)
    mean = np.squeeze(np.mean(all_rew, axis=0))
    std = np.squeeze(np.std(all_rew, axis=0))

    steps = max_steps // n_steps

    # Plot the evolution of expected rewards
    plt.plot(range(n_steps, max_steps + n_steps, n_steps), mean[:steps], label='Mean')

    # Fill between +- standard deviation
    plt.fill_between(range(n_steps, max_steps + n_steps, n_steps), mean[:steps] - std[:steps], mean[:steps] + std[:steps], alpha=0.2)

    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    # Need to specify it is Inverted Double Pendulum if double is True
    plt.title('Mean Reward over Timesteps for ' + alg.upper() + (' with Inverted Double Pendulum' if double else ' with Inverted Pendulum'))
    plt.grid(True)
    plt.savefig(alg + ('_double.png' if double else '_simple.png'))  
    
def main():
    all_algs = ["ddpg", "ppo", "sac"]
    for alg in all_algs:
        all_rew = evaluate_model(alg=alg, double=False, max_steps=max_steps)
        plot_evolution(all_rew, alg=alg, double=False)
        
        all_rew = evaluate_model(alg=alg, double=True, max_steps=max_steps)
        plot_evolution(all_rew, alg=alg, double=True)

if __name__ == '__main__':
    main()