import torch
import numpy as np
from dqn_model import DQN
from RLenviroment import TSPPlaneEnv
import argparse

def load_model(path, obs_dim, act_dim):
    model = DQN(obs_dim, act_dim)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def run_inference(model_path, num_episodes=10):
    env = TSPPlaneEnv(num_cities=3)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = load_model(model_path, obs_dim, act_dim)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            obs_tensor = torch.tensor([obs], dtype=torch.float32)
            with torch.no_grad():
                action = model(obs_tensor).argmax().item()
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1

        print(f"Episode {ep+1}: Total reward = {total_reward}, Steps = {steps}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=r'models/best.pt', help="Path to the saved .pt model file")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    args = parser.parse_args()

    run_inference(args.model, args.episodes)
