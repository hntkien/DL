"""Evaluate a Task-2/Task-3 (ALE/Pong-v5) DQN snapshot.

Runs N seeded episodes, prints per-episode reward and the mean/std,
matching the grading-protocol screenshot style.
"""

import argparse
import logging
import os
import random
import warnings
from collections import deque

import ale_py
import cv2
import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn

# Silence noisy third-party output (FFmpeg macro_block_size, torch.load FutureWarning).
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
os.environ.setdefault("IMAGEIO_FFMPEG_LOG_LEVEL", "error")

gym.register_envs(ale_py)


class DQN(nn.Module):
    """Mnih-2015 Atari CNN Q-network.

    Args:
        input_channels (int): Number of stacked grayscale frames.
        num_actions (int): Number of discrete actions.
    Input:
        x (torch.Tensor): Shape (B, input_channels, 84, 84), uint8 or float.
    Output:
        torch.Tensor: Q-values of shape (B, num_actions).
    """

    def __init__(self, input_channels, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class AtariPreprocessor:
    """Grayscale + resize to 84x84 + frame stacking for Atari."""

    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if obs.ndim == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    model = DQN(4, num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    if not args.no_video:
        os.makedirs(args.output_dir, exist_ok=True)

    # ----- Header -----
    print(f"Model:  {args.model_path}")
    print(f"Env:    ALE/Pong-v5  |  Episodes: {args.episodes}  |  "
          f"Seeds: {args.seed} to {args.seed + args.episodes - 1}")
    print("-" * 60)
    print(f"{'Episode':>7} | {'Seed':>5} | {'Reward':>8}")
    print("-" * 60)

    rewards = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        obs, _ = env.reset(seed=seed)
        state = preprocessor.reset(obs)
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
            state = preprocessor.step(next_obs)

        rewards.append(total_reward)
        print(f"{ep:>7d} | {seed:>5d} | {total_reward:>8.1f}")

        if frames is not None:
            out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
            with imageio.get_writer(out_path, fps=30, macro_block_size=1) as video:
                for f in frames:
                    video.append_data(f)

    # ----- Summary -----
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
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; episode i uses seed+i (default 0 -> grading seeds 0..19)")
    parser.add_argument("--no-video", action="store_true",
                        help="Skip saving mp4 videos for a faster, cleaner eval.")
    args = parser.parse_args()
    evaluate(args)