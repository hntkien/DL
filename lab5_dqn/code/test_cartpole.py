"""Evaluate a Task-1 (CartPole-v1) DQN snapshot.

Runs N seeded episodes, prints per-episode reward and mean/std,
and saves an mp4 of each episode (use --no-video to skip).
"""

import argparse
import logging
import os
import random
import warnings

import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
os.environ.setdefault("IMAGEIO_FFMPEG_LOG_LEVEL", "error")


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
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    if not args.no_video:
        os.makedirs(args.output_dir, exist_ok=True)

    print(f"Model:  {args.model_path}")
    print(f"Env:    {args.env_name}  |  Episodes: {args.episodes}  |  "
          f"Seeds: {args.seed} to {args.seed + args.episodes - 1}")
    print("-" * 60)
    print(f"{'Episode':>7} | {'Seed':>5} | {'Reward':>8}")
    print("-" * 60)

    rewards = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        obs, _ = env.reset(seed=seed)
        state = np.asarray(obs, dtype=np.float32)
        done = False
        total_reward = 0.0
        frames = [] if not args.no_video else None

        while not done:
            if frames is not None:
                frames.append(env.render())
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = np.asarray(next_obs, dtype=np.float32)

        rewards.append(total_reward)
        print(f"{ep:>7d} | {seed:>5d} | {total_reward:>8.1f}")

        if frames is not None:
            out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
            with imageio.get_writer(out_path, fps=30, macro_block_size=1) as video:
                for f in frames:
                    video.append_data(f)

    rewards = np.asarray(rewards, dtype=np.float64)
    print("-" * 60)
    print(f"Mean reward over {args.episodes} episodes: "
          f"{rewards.mean():.2f}  (std {rewards.std():.2f}, "
          f"min {rewards.min():.0f}, max {rewards.max():.0f})")
    if not args.no_video:
        print(f"Videos saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--env-name", type=str, default="CartPole-v1")
    parser.add_argument("--output-dir", type=str, default="./eval_cartpole")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="Base seed; episode i uses seed+i")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip saving mp4 videos for a faster, cleaner eval.")
    args = parser.parse_args()
    evaluate(args)