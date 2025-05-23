import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from dqn_model import DQNAgent
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import os
from tqdm import tqdm

from RLenviroment import TSPPlaneEnv

def make_env(num_cities_=3):
    def thunk():
        return TSPPlaneEnv(num_cities=num_cities_)
    return thunk

def train(num_envs=8, total_timesteps=100_000, save_path='saved_models'):
    os.makedirs(save_path, exist_ok=True)

    envs = AsyncVectorEnv([make_env(num_cities_=3) for _ in range(num_envs)])
    obs_space = envs.single_observation_space.shape[0]
    action_space = envs.single_action_space.n

    agent = DQNAgent(obs_dim=obs_space, act_dim=action_space)

    writer = SummaryWriter()

    obs, _ = envs.reset()
    best_reward = -float("inf")
    all_rewards = np.zeros(num_envs)

    for step in tqdm(range(total_timesteps)):
        actions = agent.act(obs)
        next_obs, rewards, dones, _, _ = envs.step(actions)

        agent.remember(obs, actions, rewards, next_obs, dones)
        agent.train_step()

        all_rewards += rewards

        for i in range(num_envs):
            if dones[i]:
                writer.add_scalar("Reward/Env_{}".format(i), all_rewards[i], step)
                if all_rewards[i] > best_reward:
                    best_reward = all_rewards[i]
                    torch.save(agent.model.state_dict(), os.path.join(save_path, "best.pt"))
                all_rewards[i] = 0

        obs = next_obs

    torch.save(agent.model.state_dict(), os.path.join(save_path, "last.pt"))
    envs.close()

if __name__ == "__main__":
    train()