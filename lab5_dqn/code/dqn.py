# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import argparse
import os
import random
import time
from collections import deque

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import yaml

gym.register_envs(ale_py)


def init_weights(m):
    """Kaiming-uniform init for Conv2d / Linear layers (ReLU-tuned)."""
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class PongActionSubsetWrapper(gym.ActionWrapper):
    """Restrict ALE/Pong-v5's 6-action set to the 3 physically meaningful ones.

    Pong's minimal action set (``full_action_space=False``) is
    [NOOP=0, FIRE=1, RIGHT=2, LEFT=3, RIGHTFIRE=4, LEFTFIRE=5]. Once the ball is
    in play, FIRE is a no-op, so RIGHTFIRE/LEFTFIRE behave identically to
    RIGHT/LEFT. This wrapper exposes only {NOOP, RIGHT, LEFT}, shrinking the
    Q-head from 6 outputs to 3 and removing the redundant action duplication
    that wastes network capacity. Matches Mnih-2015's Pong action set.

    Args:
        env (gym.Env): Underlying Gymnasium env with ALE-Pong's 6-action set.
    Input action (from agent):
        int in {0, 1, 2}: NOOP, RIGHT, LEFT respectively.
    Output action (forwarded to env):
        int in {0, 2, 3}: ALE Pong's NOOP, RIGHT, LEFT.
    """

    _SUBSET = [0, 2, 3]  # NOOP, RIGHT, LEFT

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(self._SUBSET))

    def action(self, action):
        return self._SUBSET[int(action)]


def build_env(env_name, *, render_mode="rgb_array",
              disable_sticky_actions=False, pong_action_subset=False):
    """Create a Gymnasium env, optionally applying Atari-specific wrappers.

    For ALE envs (``env_name`` starts with ``"ALE/"``), supports disabling
    the default 25%-probability sticky actions (which add extra stochasticity
    via ``repeat_action_probability``). For Pong specifically, supports
    reducing the action space to {NOOP, RIGHT, LEFT}.

    Args:
        env_name (str): Gymnasium env id (e.g. "ALE/Pong-v5", "CartPole-v1").
        render_mode (str): Forwarded to ``gym.make``.
        disable_sticky_actions (bool): If True and env is ALE/*, pass
            ``repeat_action_probability=0.0`` to ``gym.make``. Default False
            preserves the ALE v5 default of 0.25.
        pong_action_subset (bool): If True and "Pong" in env_name, wrap with
            ``PongActionSubsetWrapper``.
    Returns:
        gym.Env: Configured environment.
    """
    kwargs = {"render_mode": render_mode}
    if env_name.startswith("ALE/") and disable_sticky_actions:
        kwargs["repeat_action_probability"] = 0.0
    env = gym.make(env_name, **kwargs)
    if pong_action_subset and "Pong" in env_name:
        env = PongActionSubsetWrapper(env)
    return env


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
        super().__init__()
        self.is_image = input_dim is None
        if not self.is_image:
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions),
            )
        else:
            self.network = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(7 * 7 * 64, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions),
            )

    def forward(self, x):
        if self.is_image:
            x = x / 255.0
        return self.network(x)


class AtariPreprocessor:
    """Grayscale + resize to 84x84 + frame stacking for Atari."""

    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """Proportional Prioritized Experience Replay (Schaul et al., 2016).

    Priorities are stored already raised to the power ``alpha`` for sampling
    speed. New transitions are inserted with the current max priority so they
    are seen at least once before their TD-error is computed.

    Args:
        capacity (int): Maximum number of transitions to store.
        alpha (float): Priority exponent. 0 = uniform sampling, 1 = greedy.
        beta (float): Initial importance-sampling exponent (annealed to 1
            externally via the ``beta`` attribute).
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0  # pre-powered (alpha already applied)
        self.eps = 1e-6

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, error=None):
        """Insert a transition with priority derived from ``error``.

        Args:
            transition: Tuple ``(s, a, r, s', done)``.
            error (float | None): TD-error magnitude. If ``None`` (typical
                case at insertion time), use the current max priority so the
                transition is sampled at least once.
        """
        priority = self.max_priority if error is None else (abs(error) + self.eps) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch proportionally to priority.

        Returns:
            tuple: ``(batch, indices, weights)`` where ``batch`` is a list
            of transitions, ``indices`` is an ``np.ndarray[int64]`` of the
            sampled positions (used by ``update_priorities``), and
            ``weights`` is an ``np.ndarray[float32]`` of normalized IS
            weights.
        """
        n = len(self.buffer)
        prios = self.priorities[:n]
        probs = prios / prios.sum()
        indices = np.random.choice(n, batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # normalize for stability
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices, errors):
        """Recompute and store ``p_i = (|error_i| + eps)^alpha`` for the given indices."""
        for idx, err in zip(indices, errors):
            p = (abs(float(err)) + self.eps) ** self.alpha
            self.priorities[idx] = p
            if p > self.max_priority:
                self.max_priority = p


class NStepBuffer:
    """Rolling n-step return accumulator.

    Holds the last ``n`` 1-step transitions; emits one n-step transition per
    ``push`` once the buffer is full, and ``flush`` empties the tail at
    episode termination so transitions starting in the last n-1 steps are
    not lost.

    Args:
        n (int): Number of steps. ``n=1`` reduces to the 1-step case.
        gamma (float): Discount factor used for the partial return.
    """

    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buf = deque(maxlen=n)

    def reset(self):
        self.buf.clear()

    def push(self, transition):
        """Add a 1-step transition; return the n-step transition or ``None``."""
        self.buf.append(transition)
        if len(self.buf) < self.n:
            return None
        return self._compute(0)

    def flush(self):
        """At episode end, emit n-step transitions for the remaining tail."""
        results = []
        for start in range(1, len(self.buf)):
            results.append(self._compute(start))
        self.buf.clear()
        return results

    def _compute(self, start):
        items = list(self.buf)[start:]
        s0, a0, _, _, _ = items[0]
        R = 0.0
        last_k = len(items) - 1
        terminated = items[-1][4]
        for k, (_, _, r, _, d) in enumerate(items):
            R += (self.gamma ** k) * r
            if d:
                last_k = k
                terminated = True
                break
        _, _, _, s_last, _ = items[last_k]
        return (s0, a0, R, s_last, terminated)


class DQNAgent:
    """DQN agent with optional Double DQN, PER, and n-step return.

    Behavior reduces to vanilla DQN when ``use_ddqn=False``, ``use_per=False``,
    and ``n_step=1`` (the defaults).

    Args:
        env_name (str): Gymnasium env id.
        args (argparse.Namespace): Hyperparameters; see ``_build_parser``.
    """

    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = build_env(
            env_name,
            disable_sticky_actions=args.disable_sticky_actions,
            pong_action_subset=args.pong_action_subset,
        )
        self.test_env = build_env(
            env_name,
            disable_sticky_actions=args.disable_sticky_actions,
            pong_action_subset=args.pong_action_subset,
        )
        self.num_actions = self.env.action_space.n

        obs_shape = self.env.observation_space.shape
        if len(obs_shape) == 1:
            self.is_image = False
            self.preprocessor = None
            self.input_dim = obs_shape[0]
        else:
            self.is_image = True
            self.preprocessor = AtariPreprocessor()
            self.input_dim = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        print(f"Env: {env_name} | num_actions: {self.num_actions} | "
              f"sticky_disabled: {args.disable_sticky_actions} | "
              f"pong_action_subset: {args.pong_action_subset}")

        self.q_net = DQN(self.num_actions, self.input_dim).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions, self.input_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        # ----- Enhancement switches -----
        self.use_ddqn = args.use_ddqn
        self.use_per = args.use_per
        self.n_step = max(1, int(args.n_step))
        self.loss_type = args.loss_type
        self.per_beta_start = args.per_beta_start
        self.per_beta_anneal_steps = args.per_beta_anneal_steps

        # ----- Replay memory -----
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                capacity=args.memory_size,
                alpha=args.per_alpha,
                beta=args.per_beta_start,
            )
        else:
            self.memory = deque(maxlen=args.memory_size)

        self.nstep = NStepBuffer(self.n_step, args.discount_factor)

        # ----- Hyperparameters -----
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.gamma_n = self.gamma ** self.n_step  # for n-step bootstrap
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.grad_clip = args.grad_clip

        # ----- Training schedule -----
        self.env_count = 0
        self.train_count = 0
        self.best_reward = args.best_reward_init
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step

        # ----- Step-based snapshots (Task 3 requirement) -----
        self.snapshot_steps = sorted(args.snapshot_steps or [])
        self.snapshots_saved = set()

        # ----- First time eval reward crosses score threshold -----
        self.score_threshold = args.score_threshold
        self.first_threshold_step = None

        # ----- Evaluation -----
        self.num_eval_episodes = args.num_eval_episodes

        # ----- I/O -----
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # ========== Helpers ========== #
    def _reset_state(self, obs):
        if self.preprocessor is not None:
            return self.preprocessor.reset(obs)
        return np.asarray(obs, dtype=np.float32)

    def _step_state(self, obs):
        if self.preprocessor is not None:
            return self.preprocessor.step(obs)
        return np.asarray(obs, dtype=np.float32)

    def _push_memory(self, transition):
        if self.use_per:
            self.memory.add(transition, error=None)  # max priority on insert
        else:
            self.memory.append(transition)

    def _maybe_save_snapshots(self):
        for step in self.snapshot_steps:
            if step in self.snapshots_saved:
                continue
            if self.env_count >= step:
                path = os.path.join(self.save_dir, f"snapshot_{step}.pt")
                torch.save(self.q_net.state_dict(), path)
                self.snapshots_saved.add(step)
                print(f"Saved milestone snapshot at env step {step} -> {path}")

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    # ========== Train / Run ========== #
    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self._reset_state(obs)
            self.nstep.reset()
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self._step_state(next_obs)

                nstep_tr = self.nstep.push((state, action, reward, next_state, done))
                if nstep_tr is not None:
                    self._push_memory(nstep_tr)
                if done:
                    for tail in self.nstep.flush():
                        self._push_memory(tail)

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1
                self._maybe_save_snapshots()

                if self.env_count % 1000 == 0:
                    print(
                        f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} "
                        f"UC: {self.train_count} Eps: {self.epsilon:.4f}"
                    )
                    wandb.log(
                        {
                            "Episode": ep,
                            "Step Count": step_count,
                            "Env Step Count": self.env_count,
                            "Update Count": self.train_count,
                            "Epsilon": self.epsilon,
                        }
                    )

            print(
                f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} "
                f"UC: {self.train_count} Eps: {self.epsilon:.4f}"
            )
            wandb.log(
                {
                    "Episode": ep,
                    "Total Reward": total_reward,
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Epsilon": self.epsilon,
                    "Replay Buffer Size": len(self.memory),
                }
            )

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
                if (
                    self.first_threshold_step is None
                    and eval_reward >= self.score_threshold
                ):
                    self.first_threshold_step = self.env_count
                    score_path = os.path.join(
                        self.save_dir, f"first_score_{int(self.score_threshold)}.pt"
                    )
                    torch.save(self.q_net.state_dict(), score_path)
                    print(
                        f"*** First reached score {self.score_threshold} at env step "
                        f"{self.env_count}; saved -> {score_path}"
                    )
                    wandb.log(
                        {
                            "First reached threshold": self.score_threshold,
                            "Steps to threshold": self.env_count,
                            "Env Step Count": self.env_count,
                        }
                    )
                print(
                    f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} "
                    f"SC: {self.env_count} UC: {self.train_count}"
                )
                wandb.log(
                    {
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Eval Reward": eval_reward,
                    }
                )

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
        if len(self.memory) < max(self.replay_start_size, self.batch_size):
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        # ----- Sample batch -----
        if self.use_per:
            progress = min(1.0, self.train_count / max(1, self.per_beta_anneal_steps))
            self.memory.beta = self.per_beta_start + progress * (1.0 - self.per_beta_start)
            batch, indices, weights_np = self.memory.sample(self.batch_size)
            weights = torch.from_numpy(weights_np).to(self.device)
        else:
            batch = random.sample(self.memory, self.batch_size)
            indices = None
            weights = None

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # ----- Bellman target (DDQN or vanilla) -----
        with torch.no_grad():
            if self.use_ddqn:
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1)[0]
            # rewards already accumulated over n-step; bootstrap with gamma^n
            target_q = rewards + self.gamma_n * next_q * (1.0 - dones)

        # ----- Loss (Huber or MSE, with optional IS-weights) -----
        if self.loss_type == "huber":
            elementwise = F.smooth_l1_loss(q_values, target_q, reduction="none")
        else:
            elementwise = F.mse_loss(q_values, target_q, reduction="none")
        if weights is not None:
            loss = (weights * elementwise).mean()
        else:
            loss = elementwise.mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()

        # ----- Update PER priorities with new TD-errors -----
        if self.use_per:
            td_errors = (target_q - q_values).detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors)

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            log = {
                "Loss": loss.item(),
                "Q mean": q_values.mean().item(),
                "Q std": q_values.std().item(),
                "Update Count": self.train_count,
                "Env Step Count": self.env_count,
            }
            if self.use_per:
                log["PER beta"] = self.memory.beta
                log["PER mean weight"] = float(weights_np.mean())
            print(
                f"[Train #{self.train_count}] Loss: {loss.item():.4f} "
                f"Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}"
            )
            wandb.log(log)


# ===================== CLI + YAML config plumbing ===================== #
def _build_parser(parents=()):
    parser = argparse.ArgumentParser(parents=list(parents))
    # Env / I/O
    parser.add_argument("--env_name", type=str, default="CartPole-v1")
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--wandb_project", type=str, default="DLP-Lab5-DQN-CartPole")
    parser.add_argument("--wandb_run_name", type=str, default="cartpole-run")
    parser.add_argument("--episodes", type=int, default=1000)
    # Optimization / replay
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--memory_size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    # Exploration
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_decay", type=float, default=0.9995)
    parser.add_argument("--epsilon_min", type=float, default=0.05)
    # Training schedule
    parser.add_argument("--target_update_frequency", type=int, default=500)
    parser.add_argument("--replay_start_size", type=int, default=1000)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--train_per_step", type=int, default=1)
    # Evaluation
    parser.add_argument("--num_eval_episodes", type=int, default=5)
    parser.add_argument("--best_reward_init", type=float, default=0.0)
    # Enhancements (Task 3)
    parser.add_argument("--use_ddqn", action="store_true", default=False)
    parser.add_argument("--use_per", action="store_true", default=False)
    parser.add_argument("--n_step", type=int, default=1)
    parser.add_argument("--per_alpha", type=float, default=0.6)
    parser.add_argument("--per_beta_start", type=float, default=0.4)
    parser.add_argument("--per_beta_anneal_steps", type=int, default=1_000_000)
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "huber"])
    parser.add_argument(
        "--score_threshold", type=float, default=1e9,
        help="Save snapshot the first time eval reward reaches this score (set <= achievable to enable).",
    )
    parser.add_argument(
        "--snapshot_steps", type=int, nargs="*", default=None,
        help="Env-step milestones at which to save snapshots (e.g. 600000 1000000 ...).",
    )
    # Atari env-side wrappers
    parser.add_argument(
        "--disable_sticky_actions", action="store_true", default=False,
        help="Pass repeat_action_probability=0.0 for ALE envs to disable sticky actions.",
    )
    parser.add_argument(
        "--pong_action_subset", action="store_true", default=False,
        help="For Pong, restrict the action space to {NOOP, RIGHT, LEFT} (3 actions).",
    )
    return parser


def parse_args():
    """Two-stage parsing: load YAML defaults (if any), then apply CLI overrides."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    yaml_cfg = {}
    if pre_args.config is not None:
        if not os.path.exists(pre_args.config):
            raise FileNotFoundError(f"Config file not found: {pre_args.config}")
        with open(pre_args.config, "r") as f:
            yaml_cfg = yaml.safe_load(f) or {}

    parser = _build_parser(parents=[pre])
    parser.set_defaults(**yaml_cfg)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        save_code=True,
        config=vars(args),
    )
    # Make Env Step Count the default x-axis for every metric in this run.
    # Existing runs without this still expose Env Step Count as a metric and
    # can be re-axised in the W&B panel UI.
    wandb.define_metric("Env Step Count")
    wandb.define_metric("*", step_metric="Env Step Count")

    agent = DQNAgent(env_name=args.env_name, args=args)
    agent.run(episodes=args.episodes)