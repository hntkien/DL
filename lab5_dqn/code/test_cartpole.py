"""Evaluate a Task-1 (CartPole-v1) DQN snapshot.

Runs N episodes with seeded resets, prints per-episode reward and the mean,
matching the grading protocol used for Pong in test_model.py.
"""

import argparse
import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    """MLP Q-network matching the architecture trained in dqn.py for CartPole.

    Args:
        input_dim (int): State dimensionality (4 for CartPole-v1).
        num_actions (int): Number of discrete actions (2 for CartPole-v1).
    Input:
        x (torch.Tensor): Shape (B, input_dim).
    Output:
        torch.Tensor: Q-values, shape (B, num_actions).
    """

    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return self.network(x)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env_name, render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    input_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model = DQN(input_dim, num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = np.asarray(obs, dtype=np.float32)
        done = False
        total_reward = 0.0
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = np.asarray(next_obs, dtype=np.float32)
        rewards.append(total_reward)
        print(f"Seed: {args.seed + ep}, eval reward: {total_reward}")

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    print(f"Average reward over {args.episodes} episodes: {mean_r:.2f} (+/- {std_r:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--env-name", type=str, default="CartPole-v1")
    parser.add_argument("--output-dir", type=str, default="./eval_cartpole")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="Base seed; episode i uses seed+i")
    args = parser.parse_args()
    evaluate(args)