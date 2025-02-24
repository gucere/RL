import numpy as np
import hockey.hockey_env as h_env
import gymnasium as gym
from importlib import reload
import time
import pylab as plt
import random
#imports needed for TD3
from emre_version.TD3 import TD3
from emre_version.feedforward import Feedforward
from emre_version.memory import Memory
from emre_version.TD3_alternative import TD3 as TD3_Alternative

# Training loop for TD3
def train_td3(agent, env, episodes=10, max_steps=1000):
    rewards = []
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.act(obs)  # TD3 decides action
            next_obs, reward, done, trunc, _ = env.step(action)
            agent.store_transition((obs, action, reward, next_obs, float(done)))
            agent.train()  # Train the TD3 agent
            episode_reward += reward
            obs = next_obs
            if done or trunc:
                break
        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
    return rewards

# Evaluation loop for TD3
def evaluate_td3(agent, env, episodes=10, max_steps=1000):
    total_rewards = []
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            action = agent.act(obs, eps=0.0)  # Disable exploration during evaluation
            next_obs, reward, done, trunc, _ = env.step(action)
            episode_reward += reward
            obs = next_obs
            if done or trunc:
                break
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward}")
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {episodes} episodes: {avg_reward}")
    return avg_reward

env = h_env.HockeyEnv()
#instance of TD3
td3_agent = TD3(
    observation_space=env.observation_space,
    action_space=env.action_space,
    userconfig={
            # "eps": 0.1,            # Epsilon: noise strength to add to policy
            # "discount": 0.95,
            # "buffer_size": int(1e6),
            # "batch_size": 128,
            # "learning_rate_actor": 0.00001,
            # "learning_rate_critic": 0.0001,
            # "hidden_sizes_actor": [128,128],
            # "hidden_sizes_critic": [128,128,64],
            # "update_target_every": 100,
            # "use_target_net": True,
            "policy_noise": 0.1,
		    "noise_clip": 0.5,
            # "policy_freq": 2, 
            # "tau": 0.005
    }
)
train_rewards = train_td3(td3_agent, env, episodes=10)
evaluate_td3(td3_agent, env, episodes=10)
plt.plot(train_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('TD3 Training Performance')
plt.show()