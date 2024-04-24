from selector import selector

import torch
import gymnasium as gym

import time

NUM_ACTIONS = 100
ACTIONS = torch.linspace(-3, 3, NUM_ACTIONS)

class FQIAgent:
    def __init__(self, model):
        self.model = model

    def get_action(self, state, actions=ACTIONS):
        ''' Returns the optimal action for a given state. '''
        with torch.no_grad():

            state = torch.tensor(state, dtype=torch.float32)
            state = state.unsqueeze(0)
            state = self.set_z(state)

            q_values = self.model(state)
            max_q_i = torch.argmax(q_values)

            action = ACTIONS[max_q_i]
        return action.item()
    
    def set_z(self, states, actions=ACTIONS):
        # Number of actions and batch size
        num_actions = actions.size(0)
        batch_size = states.size(0)
        
        # Repeat each state for each action
        states = states.unsqueeze(1).repeat(1, num_actions, 1)
        states = states.view(batch_size * num_actions, -1)

        # Repeat actions for the whole batch
        actions = actions.repeat(batch_size, 1).view(batch_size * num_actions, 1)
        
        # Concatenate states with actions
        state_action_pairs = torch.cat((states, actions), dim=1)
        
        return state_action_pairs


def inv_fqi(env, num_features):
    NUM_ACTIONS = 100
    ACTIONS = torch.linspace(-3, 3, NUM_ACTIONS)

    model = QNetwork(input_size=num_features+1, output_size=1)
    file = "./models_final/fqi.pth"
    model.load_state_dict(torch.load(file))
    model.eval()

    agent = FQIAgent(model)

    observation, info = env.reset()
    done = False
    rewards = 0
    while not done:
        action = agent.get_action(observation, actions=ACTIONS)
        next_observation, reward, terminated, truncated, info = env.step([action])
        rewards += reward
        done = terminated or truncated
        observation = next_observation  
    print(f"Total Reward earned with FQI: {rewards}")

def inv_reinforce(env, num_features):
    model = GaussianFeedForward(input_size=num_features, output_size=2)
    file = "./models_final/reinforce.pth"
    model.load_state_dict(torch.load(file))
    model.eval()

    observation, info = env.reset()
    done = False
    rewards = 0
    while not done:
        observation = torch.tensor(observation, dtype=torch.float32)

        out = model(observation)
        mu, logvar = out[0], out[1]

        action, _ = sample_action(mu, logvar)
        
        action = torch.clamp(action, -3, 3)

        next_observation, reward, terminated, truncated, info = env.step([action])
        
        rewards += reward
        done = terminated or truncated
        observation = next_observation
    print(f"Total Reward earned with REINFORCE: {rewards}")

def inv_ddpg(env, num_features):

    actor = Actor(num_features, 1)
    file = "./models_final/actor.pth"
    actor.load_state_dict(torch.load(file))

    state, _ = env.reset()
    total_rew = 0
    done = False
    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        action = actor(state).detach()
        next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        total_rew += reward

        done = terminated or truncated

        state = next_state

    print(f"Total reward: {total_rew}")



def double_inv_fqi(env, num_features):
    NUM_ACTIONS = int(100/3)
    ACTIONS = torch.linspace(-1, 1, NUM_ACTIONS)

    model = QNetwork(input_size=num_features+1, output_size=1)
    file = "./models_final/dfqi.pth"
    model.load_state_dict(torch.load(file))
    model.eval()

    agent = FQIAgent(model)

    observation, info = env.reset()
    done = False
    rewards = 0
    while not done:
        action = agent.get_action(observation, actions=ACTIONS)
        next_observation, reward, terminated, truncated, info = env.step([action])
        rewards += reward
        done = terminated or truncated
        observation = next_observation  
    print(f"Total Reward earned with FQI: {rewards}")

def double_inv_reinforce(env, num_features):
    model  = GaussianFeedForward(input_size=num_features, output_size=2, do_dropout=False, hidden_sizes=[48, 32])
    file = "./models_final/dreinforce.pth"
    model.load_state_dict(torch.load(file))
    model.eval()

    observation, info = env.reset()
    done = False
    rewards = 0
    while not done:
        observation = torch.tensor(observation, dtype=torch.float32)

        out = model(observation)
        mu, logvar = out[0], out[1]

        action, _ = sample_action(mu, logvar)
        
        action = torch.clamp(action, -1, 1)

        next_observation, reward, terminated, truncated, info = env.step([action])
        
        rewards += reward
        done = terminated or truncated
        if done and rewards < 1000:
            rewards = 0
            observation, info = env.reset()
            done = False
            
        observation = next_observation

    print(f"Total Reward earned with REINFORCE: {rewards}")

def double_inv_ddpg(env, num_features):
    actor = Actor(num_features, 1, double=True)
    file = "./models_final/dactor.pth"
    actor.load_state_dict(torch.load(file))

    state, _ = env.reset()
    total_rew = 0
    done = False
    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        action = actor(state).detach()
        next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        total_rew += reward

        done = terminated or truncated
        state = next_state

    print(f"Total reward: {total_rew}")


if __name__ == "__main__":
    sel = selector()

    simu = sel[0]
    algo = sel[1]

    if simu == 'Inverted Pendulum':
        
        env = 'InvertedPendulum-v4'
        env = gym.make(env, render_mode='human')
        num_features = env.observation_space.shape[0]

        if algo == 'FQI':
            from fqi import QNetwork
            inv_fqi(env, num_features)
        elif algo == 'REINFORCE':
            from reinforce import GaussianFeedForward, sample_action
            inv_reinforce(env, num_features)
        elif algo == 'DDPG':
            from ddpg import Actor
            inv_ddpg(env, num_features)
        else:
            raise ValueError("Invalid Algorithm")

    elif simu == 'Double Inverted Pendulum':
        env = 'InvertedDoublePendulum-v4'
        env = gym.make(env, render_mode='human')
        num_features = env.observation_space.shape[0]

        if algo == 'FQI':
            from fqi import QNetwork
            double_inv_fqi(env, num_features)
        elif algo == 'REINFORCE':
            from reinforce import GaussianFeedForward, sample_action
            double_inv_reinforce(env, num_features)
        elif algo == 'DDPG':
            from ddpg import Actor
            double_inv_ddpg(env, num_features)
        else:   
            raise ValueError("Invalid Algorithm")
    else:
        raise ValueError("Invalid Environment")
    

