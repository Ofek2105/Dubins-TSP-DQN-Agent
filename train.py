import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from dqn_model import DQNAgent
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import os
from tqdm import tqdm

from RLenviroment import TSPPlaneEnv

def make_env(num_cities_=3, max_steps=500):
    def thunk():
        return TSPPlaneEnv(num_cities=num_cities_, max_steps=500)
    return thunk

def train(num_envs=8, total_timesteps=200_000, save_pt_path='saved_models'):
    os.makedirs(save_pt_path, exist_ok=True)

    env_list = [make_env(num_cities_=3, max_steps=500),
                make_env(num_cities_=5, max_steps=500),
                make_env(num_cities_=10, max_steps=700),
                make_env(num_cities_=5, max_steps=300),
                make_env(num_cities_=2, max_steps=500),
                make_env(num_cities_=1, max_steps=200),
                make_env(num_cities_=5, max_steps=200),
                make_env(num_cities_=8, max_steps=500)]

    envs = AsyncVectorEnv(env_list)
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

        # writer.add_scalar("Reward_AVG", np.mean(all_rewards), step)
        for i in range(num_envs):
            if dones[i]:
                writer.add_scalar("Reward/Env_{}".format(i), all_rewards[i], step)
                if all_rewards[i] > best_reward:
                    best_reward = all_rewards[i]
                    torch.save(agent.model.state_dict(), os.path.join(save_pt_path, "best.pt"))
                all_rewards[i] = 0

        obs = next_obs

    torch.save(agent.model.state_dict(), os.path.join(save_pt_path, "last.pt"))
    envs.close()

if __name__ == "__main__":
    train()