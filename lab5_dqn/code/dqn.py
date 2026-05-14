# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """Deep Q-Network with two modes.
 
    - MLP mode (``input_dim`` is an int): a 2-hidden-layer fully connected net,
      used for low-dimensional state spaces such as CartPole-v1.
    - CNN mode (``input_dim`` is None): the Mnih-2015 Atari architecture
      (Conv 8x8/s4 -> 4x4/s2 -> 3x3/s1 -> FC 512 -> FC num_actions), used for
      stacked 84x84 frames in Atari Pong.
 
    Args:
        num_actions (int): Number of discrete actions in the env.
        input_dim (int | None): If int, build an MLP with this input width.
            If None, build the Atari CNN expecting input shape (4, 84, 84).
 
    Input:
        x (torch.Tensor): States. Shape (B, input_dim) for MLP mode, or
            (B, 4, 84, 84) for CNN mode.
    Output:
        torch.Tensor: Q-values of shape (B, num_actions).
    """
    def __init__(self, num_actions, input_dim=None):
        super(DQN, self).__init__()
        self.is_image = input_dim is None
        ########## YOUR CODE HERE (5~10 lines) ##########
        if not self.is_image:
            # Task 1: MLP for CartPole (state is a 4-D vector)
            self.network = nn.Sequential(
                nn.Linear(in_features=input_dim, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=num_actions)
            )
        else:
            # Task 2/3: Classic Atari CNN 
            self.network = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(in_features=7*7*64, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=num_actions)
            )
        
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        # Normliaze pixel intensitites to [0,1] for CNN mode
        if self.is_image:
            x = x / 255.0
        return self.network(x)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ########## 
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
        

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        # Detect state type 
        obs_shape = self.env.observation_space.shape 
        if len(obs_shape) == 1:
            # Vector state (e.g., CartPole)
            self.is_image = False 
            self.preprocessor = None 
            self.input_dim = obs_shape[0]
        else:
            # Image state (e.g., Atari Pong)
            self.is_image = True
            self.preprocessor = AtariPreprocessor()
            self.input_dim = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)


        self.q_net = DQN(self.num_actions, self.input_dim).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions, self.input_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        # Uniform sampling replay buffer (task 1/2)
        self.memory = deque(maxlen=args.memory_size)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = args.best_reward_init
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # ========== Helper Functions ========== #
    def _reset_state(self, obs):
        if self.preprocessor is not None:
            return self.preprocessor.reset(obs)
        return np.asarray(obs, dtype=np.float32)
    
    def _step_state(self, obs):
        if self.preprocessor is not None:
            return self.preprocessor.step(obs)
        return np.asarray(obs, dtype=np.float32)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self._reset_state(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state = self._step_state(next_obs)
                self.memory.append((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            wandb.log({
                "Replay Buffer Size": len(self.memory),
                "Env Step Count": self.env_count,
            })
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def evaluate(self):
        total = 0.0
        for _ in range(self.num_eval_episodes):
            obs, _ = self.test_env.reset()
            state = self._reset_state(obs)
            done = False
            ep_reward = 0
            while not done:
                state_tensor = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                ep_reward += reward
                state = self._step_state(next_obs)
            total += ep_reward
        return total / self.num_eval_episodes


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q_max * (1.0 - dones)
 
        # MSE loss between predicted Q(s,a) and the bootstrapped target.
        loss = F.mse_loss(q_values, target_q)
 
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
           print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")

           wandb.log({
                "Loss": loss.item(),
                "Q mean": q_values.mean().item(),
                "Q std": q_values.std().item(),
                "Update Count": self.train_count,
                "Env Step Count": self.env_count,
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Env / I/O.
    parser.add_argument("--env_name", type=str, default="CartPole-v1")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--wandb_project", type=str, default="DLP-Lab5-DQN-CartPole")
    parser.add_argument("--wandb_run_name", type=str, default="cartpole-run")
    parser.add_argument("--episodes", type=int, default=1000)
    # Optimization / replay.
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--memory_size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--grad_clip", type=float, default=10.0,
                        help="Max L2 norm for gradient clipping; <=0 disables.")
    # Exploration.
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_decay", type=float, default=0.999999,
                        help="Multiplicative decay applied once per training update.")
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    # Training schedule.
    parser.add_argument("--target_update_frequency", type=int, default=1000)
    parser.add_argument("--replay_start_size", type=int, default=50000)
    parser.add_argument("--max_episode_steps", type=int, default=10000)
    parser.add_argument("--train_per_step", type=int, default=1)
    # Evaluation.
    parser.add_argument("--num_eval_episodes", type=int, default=5,
                        help="Number of episodes averaged per evaluate() call during training.")
    parser.add_argument("--best_reward_init", type=float, default=0.0,
                        help="Initial threshold for saving best_model.pt (0 for CartPole, -21 for Pong).")
    args = parser.parse_args()
 
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, save_code=True, config=vars(args))
    agent = DQNAgent(env_name=args.env_name, args=args)
    agent.run(episodes=args.episodes)