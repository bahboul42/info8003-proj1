from selector import selector
from fqi import QNetwork
from reinforce import GaussianFeedForward, sample_action

import torch
import gymnasium as gym

import time

if __name__ == "__main__":
    sel = selector()

    simu = sel[0]
    algo = sel[1]

    isDouble = False

    if simu == 'Inverted Pendulum':
        
        env = 'InvertedPendulum-v4'
        env = gym.make(env, render_mode='human')
        num_features = env.observation_space.shape[0]
        if algo == 'FQI':
            model = QNetwork(input_size=num_features, output_size=1)
        elif algo == 'REINFORCE':
            model = GaussianFeedForward(input_size=num_features, output_size=2)
            file = "./models/reinforce_4700-42.pth"
        elif algo == 'PPO':
            pass
        else:
            raise ValueError("Invalid Algorithm")
        

    elif simu == 'Double Inverted Pendulum':
        isDouble = True
        env = 'DoubleInvertedPendulum-v4'
    else:
        raise ValueError("Invalid Environment")
    


    
    time.sleep(.3)

    model.load_state_dict(torch.load(file))
    model.eval()

    print("Max episode steps:", env.spec.max_episode_steps)
    observation, info = env.reset()
    done = False
    
    i = 0
    while not done:
        i+=1
        if i == 1000:
            print("Max steps reached")
        observation = torch.tensor(observation, dtype=torch.float32)

        out = model(observation)

        if algo == 'REINFORCE':
            mu, logvar = out[0], out[1]
            action, _ = sample_action(mu, logvar)
        
        action = torch.clamp(action, -1, 1) if isDouble else torch.clamp(action, -3, 3)

        next_observation, reward, terminated, truncated, info = env.step([action])
        
        done = terminated or truncated
        observation = next_observation

    print("Starting Simulation...")