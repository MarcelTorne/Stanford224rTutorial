"""Quick demo: three approaches to bimodal imitation learning (action chunks).

Part 1 — Hard mode (two gaps): BC regression averages bimodal expert actions
and crashes; Diffusion policy models the full distribution and succeeds.

Part 2 — Hard mode (DAgger): starting from the same bimodal data, DAgger
relabels with a deterministic expert (always upper gap), gradually resolving
the ambiguity so BC regression succeeds.

Action chunking: policies predict ACTION_CHUNK future target positions at once.
During rollout, only the first ACTION_CHUNK//2 are executed before re-predicting
(receding horizon).  Videos visualise all predicted targets as dots.

Usage:
    srun --reservation=daily --gres=gpu:1 --pty bash
    conda activate flow-dpo
    python quick_demo_chunks_multi_seed.py

"""

import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from typing import Union
from collect_data import Expert, compute_action, compute_action_gravity, COMMIT_DIST
from flappy_bird_env import FlappyBirdEnv, SCREEN_W, SCREEN_H, BIRD_X
from train_bc import BCPolicy
from train_diffusion import DDPMSchedule

# ---------------------------------------------------------------------------
# 1-D Temporal U-Net  (adapted from Chi et al., Diffusion Policy)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size,
                      padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim,
                 kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size,
                        n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size,
                        n_groups=n_groups),
        ])
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
            nn.Unflatten(-1, (-1, 1)),
        )
        self.residual_conv = (nn.Conv1d(in_channels, out_channels, 1)
                              if in_channels != out_channels
                              else nn.Identity())

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale, bias = embed[:, 0], embed[:, 1]
        out = scale * out + bias
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim,
                 diffusion_step_embed_dim=32,
                 down_dims=(32, 64), kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim,
                                       kernel_size=kernel_size,
                                       n_groups=n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim,
                                       kernel_size=kernel_size,
                                       n_groups=n_groups),
        ])

        down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out,
                                           cond_dim=cond_dim,
                                           kernel_size=kernel_size,
                                           n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out,
                                           cond_dim=cond_dim,
                                           kernel_size=kernel_size,
                                           n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in,
                                           cond_dim=cond_dim,
                                           kernel_size=kernel_size,
                                           n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in,
                                           cond_dim=cond_dim,
                                           kernel_size=kernel_size,
                                           n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.down_modules = down_modules
        self.up_modules = up_modules
        self.final_conv = final_conv

    def forward(self, sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                global_cond=None):
        sample = sample.moveaxis(-1, -2)  # (B,T,C) -> (B,C,T)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long,
                                     device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond],
                                       dim=-1)

        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)  # (B,C,T) -> (B,T,C)


class TemporalNoisePredictor(nn.Module):
    """Wraps ConditionalUnet1D to match the (B, action_dim) interface.

    Reshapes the flat action vector (B, K) into (B, K, 1) for the U-Net,
    and flattens the output back to (B, K).
    """

    def __init__(self, state_dim=4, pred_horizon=20, action_dim=1,
                 **unet_kwargs):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=state_dim,
            **unet_kwargs,
        )

    def forward(self, noisy_action, state, timestep):
        B = noisy_action.shape[0]
        x = noisy_action.view(B, self.pred_horizon, self.action_dim)
        out = self.unet(x, timestep, global_cond=state)
        return out.reshape(B, -1)


# ---------------------------------------------------------------------------
# Gaussian BC Policy (learned mean + variance)
# ---------------------------------------------------------------------------

class GaussianBCPolicy(nn.Module):
    """MLP that outputs mean and log-variance for a Gaussian over actions."""

    def __init__(self, state_dim=4, action_dim=1, hidden=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(hidden, action_dim),
            nn.Sigmoid(),
        )
        self.log_var_head = nn.Linear(hidden, action_dim)

    def forward(self, state):
        h = self.backbone(state)
        mean = self.mean_head(h)
        log_var = self.log_var_head(h)
        return mean, log_var

    def sample(self, state):
        mean, log_var = self.forward(state)
        std = (0.5 * log_var).exp()
        return (mean + std * torch.randn_like(std)).clamp(0, 1)

    def deterministic(self, state):
        mean, _ = self.forward(state)
        return mean


class GaussianWrapper:
    """Wrap GaussianBCPolicy for evaluate_policy (det or stochastic)."""

    def __init__(self, model, stochastic=False):
        self.model = model
        self.stochastic = stochastic

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, state):
        if self.stochastic:
            return self.model.sample(state)
        return self.model.deterministic(state)

    def state_dict(self):
        return self.model.state_dict()


# ---------------------------------------------------------------------------
# Deterministic expert for DAgger relabeling
# ---------------------------------------------------------------------------

class DeterministicExpert:
    """Expert that always picks gap1 (upper gap) in hard mode.

    Mimics the real Expert's behavior (hover at midpoint, commit when close,
    EMA smoothing) but replaces the random gap choice with a deterministic
    one.  This produces smooth, consistent relabeling that is compatible
    with the original expert data.
    """

    def __init__(self, commit_dist: float = COMMIT_DIST, smoothing: float = 0.15):
        self.commit_dist = commit_dist
        self.smoothing = smoothing
        self._last_gap_sig = None
        self._committed = False
        self._smooth_target = None

    def reset(self):
        self._last_gap_sig = None
        self._committed = False
        self._smooth_target = None

    def act(self, obs: np.ndarray) -> float:
        dist = obs[0]
        gap1_y = obs[1]
        gap2_y = obs[2]

        gap_sig = (round(gap1_y, 3), round(gap2_y, 3))
        if self._last_gap_sig != gap_sig:
            self._committed = False
            self._last_gap_sig = gap_sig

        midpoint = (gap1_y + gap2_y) / 2.0

        if not self._committed:
            if dist < self.commit_dist:
                self._committed = True
                raw_target = float(gap1_y)
            else:
                raw_target = float(midpoint)
        else:
            raw_target = float(gap1_y)

        if self._smooth_target is None:
            self._smooth_target = raw_target
        else:
            self._smooth_target += self.smoothing * (
                raw_target - self._smooth_target)

        return float(np.clip(self._smooth_target, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_DIFFUSION_ITERS = 20
PIPE_SPEED = 3.0
ACTION_CHUNK = 20
EXECUTE_STEPS = ACTION_CHUNK // 2

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"Using device: {DEVICE}")
print(f"Action chunk: {ACTION_CHUNK}, execute first {EXECUTE_STEPS}")


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_expert_data(difficulty, num_episodes, pipe_speed=PIPE_SPEED, seed=0):
    """Collect expert demos step-by-step, then window into action chunks.

    Training pairs are (s_t, [a_t, a_{t+1}, ..., a_{t+K-1}]) where K=ACTION_CHUNK.
    """
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed)
    expert = Expert()
    all_states, all_actions = [], []
    all_steps = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        expert.reset()
        done = False
        ep_states, ep_actions = [], []
        while not done:
            action = expert.act(obs, difficulty)
            ep_states.append(obs.copy())
            ep_actions.append(action)
            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            done = terminated or truncated
        all_steps.append(len(ep_states))
        for i in range(len(ep_states) - ACTION_CHUNK + 1):
            all_states.append(ep_states[i])
            all_actions.append(ep_actions[i:i + ACTION_CHUNK])
    print(f"Average steps: {np.mean(all_steps):.1f}")
    print(f"Chunk={ACTION_CHUNK}: {len(all_states)} chunk pairs from {num_episodes} episodes")
    env.close()
    return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.float32)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_bc_policy(states, actions, epochs=50, batch_size=2048, lr=1e-3,
                    verbose=False):
    """Train BC policy that outputs ACTION_CHUNK actions."""
    s_tensor = torch.tensor(states, dtype=torch.float32)
    a_tensor = torch.tensor(actions, dtype=torch.float32)
    action_dim = a_tensor.shape[1]
    loader = DataLoader(TensorDataset(s_tensor, a_tensor),
                        batch_size=batch_size, shuffle=True)

    policy = BCPolicy(action_dim=action_dim).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(DEVICE)
            a_batch = a_batch.to(DEVICE)
            pred = policy(s_batch)
            loss = loss_fn(pred, a_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * s_batch.size(0)
            n += s_batch.size(0)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n:.6f}")

    return policy


def train_gaussian_bc_policy(states, actions, epochs=50, batch_size=2048,
                              lr=1e-3, verbose=False):
    """Train Gaussian BC policy (mean + learned variance) with NLL loss."""
    s_tensor = torch.tensor(states, dtype=torch.float32)
    a_tensor = torch.tensor(actions, dtype=torch.float32)
    action_dim = a_tensor.shape[1]
    loader = DataLoader(TensorDataset(s_tensor, a_tensor),
                        batch_size=batch_size, shuffle=True)

    policy = GaussianBCPolicy(action_dim=action_dim).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(DEVICE)
            a_batch = a_batch.to(DEVICE)
            mean, log_var = policy(s_batch)
            loss = 0.5 * (log_var + (a_batch - mean).pow(2) / log_var.exp()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * s_batch.size(0)
            n += s_batch.size(0)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n:.6f}")

    return policy


def train_diffusion_policy(states, actions, epochs=100, batch_size=2048,
                           lr=1e-4, T=100, beta_start=0.0001, beta_end=0.02,
                           verbose=False):
    """Train a DDPM diffusion policy outputting ACTION_CHUNK actions."""
    s_tensor = torch.tensor(states, dtype=torch.float32)
    a_tensor = torch.tensor(actions, dtype=torch.float32)
    action_dim = a_tensor.shape[1]
    loader = DataLoader(TensorDataset(s_tensor, a_tensor),
                        batch_size=batch_size, shuffle=True)

    model = TemporalNoisePredictor(
        state_dim=s_tensor.shape[1],
        pred_horizon=action_dim,
        action_dim=1,
    ).to(DEVICE)
    schedule = DDPMSchedule(T=T, beta_start=beta_start, beta_end=beta_end,
                            device=DEVICE, action_dim=action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(DEVICE)
            a_batch = a_batch.to(DEVICE)
            bsz = s_batch.size(0)

            t = torch.randint(0, T, (bsz,), device=DEVICE)
            noisy_a, noise = schedule.q_sample(a_batch, t)
            pred_noise = model(noisy_a, s_batch, t)
            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bsz
            n += bsz

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n:.6f}")

    return DiffusionWrapper(model, schedule)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class ExpertWrapper:
    """Wrap Expert so it has the same call interface as a learned policy.

    Returns a single action (not a chunk) — evaluated step-by-step.
    """

    def __init__(self, difficulty, env=None):
        self.expert = Expert()
        self.difficulty = difficulty
        self.env = env

    def eval(self):
        return self

    def reset(self):
        self.expert.reset()

    def set_env(self, env):
        self.env = env

    def __call__(self, state_t):
        obs = state_t.cpu().numpy()[0]
        a = self.expert.act(obs, self.difficulty)
        return torch.tensor([[a]], dtype=torch.float32, device=state_t.device)


class DiffusionWrapper:
    """Wrap (model, schedule) so it has the same call interface as BCPolicy."""

    def __init__(self, model, schedule):
        self.model = model
        self.schedule = schedule

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, state):
        return self.schedule.sample(self.model, state)

    def state_dict(self):
        return self.model.state_dict()


# ---------------------------------------------------------------------------
# Chunk executor (receding horizon)
# ---------------------------------------------------------------------------

class ChunkExecutor:
    """Execute action chunks with a receding horizon.

    Every EXECUTE_STEPS steps the policy is queried for a new chunk of
    ACTION_CHUNK actions.  Only the first EXECUTE_STEPS are executed before
    re-querying.
    """

    def __init__(self, chunk_size=ACTION_CHUNK, execute_steps=EXECUTE_STEPS):
        self.chunk_size = chunk_size
        self.execute_steps = execute_steps
        self.action_chunk = None
        self.step_in_chunk = 0

    def reset(self):
        self.action_chunk = None
        self.step_in_chunk = 0

    def needs_query(self):
        return self.action_chunk is None or self.step_in_chunk >= self.execute_steps

    def set_chunk(self, chunk):
        """chunk: numpy array or list of length ACTION_CHUNK."""
        self.action_chunk = np.array(chunk, dtype=np.float32).flatten()
        self.step_in_chunk = 0

    def get_action(self):
        action = self.action_chunk[self.step_in_chunk]
        self.step_in_chunk += 1
        return float(np.clip(action, 0.0, 1.0))

    def get_all_targets(self):
        """Return the full predicted chunk (for visualisation)."""
        if self.action_chunk is None:
            return np.array([])
        return self.action_chunk.copy()

    def current_index(self):
        return self.step_in_chunk


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def _draw_chunk_overlay(frame, chunk_targets, current_idx):
    """Draw predicted chunk positions on a rendered frame.

    - Dots are drawn at the bird's x-position, spaced slightly to the right
      for future steps.
    - Current action: bright green circle.
    - Future actions: blue circles that fade with distance.
    - Already-executed actions: dim gray.
    """
    if len(chunk_targets) == 0:
        return frame
    frame = frame.copy()
    h, w = frame.shape[:2]
    n = len(chunk_targets)
    spacing = min(8, (w - BIRD_X - 20) // max(n, 1))

    for i, target_y in enumerate(chunk_targets):
        py = int(np.clip(target_y * SCREEN_H, 0, SCREEN_H - 1))
        px = BIRD_X + 20 + i * spacing
        if px >= w:
            break
        r = 4 if i == current_idx else 3

        if i < current_idx:
            color = np.array([100, 100, 100], dtype=np.uint8)
        elif i == current_idx:
            color = np.array([60, 255, 60], dtype=np.uint8)
        else:
            fade = max(0.2, 1.0 - (i - current_idx) / n)
            color = np.array([int(60 * fade), int(120 * fade), int(255 * fade)],
                             dtype=np.uint8)

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    yy, xx = py + dy, px + dx
                    if 0 <= yy < h and 0 <= xx < w:
                        frame[yy, xx] = color

    # Draw a thin vertical line at EXECUTE_STEPS boundary
    boundary_x = BIRD_X + 20 + EXECUTE_STEPS * spacing
    if boundary_x < w:
        for y in range(h):
            frame[y, boundary_x] = [255, 255, 0]

    return frame


# ---------------------------------------------------------------------------
# Evaluation (with optional video from the same rollouts)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_policy(policy, difficulty, num_episodes, pipe_speed=PIPE_SPEED,
                    seed=100, use_chunks=True, video_path=None,
                    video_episodes=3):
    """Evaluate policy and return (mean, std) of episode lengths.

    If *use_chunks* is True, uses ChunkExecutor (for learned policies).
    If False, runs step-by-step (for the expert baseline).
    If *video_path* is given, the first *video_episodes* episodes are recorded.
    """
    import pygame

    recording = video_path is not None
    render_mode = "rgb_array" if recording else None

    policy.eval()
    if recording:
        pygame.init()
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed,
                        render_mode=render_mode)
    if hasattr(policy, "set_env"):
        policy.set_env(env)
    executor = ChunkExecutor() if use_chunks else None
    episode_lengths = []
    all_frames = []
    episode_outcomes = []
    t_eval_start = time.time()
    t_policy_total = 0.0
    t_render_total = 0.0
    n_policy_calls = 0

    for ep in range(num_episodes):
        if recording and ep == video_episodes:
            env.close()
            pygame.quit()
            if all_frames:
                os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
                iio.imwrite(video_path, np.stack(all_frames), fps=30)
                outcomes_str = ", ".join(
                    [f"Ep{i+1}: {steps}steps ({outcome})"
                     for i, (outcome, steps) in enumerate(episode_outcomes)])
                print(f"  Saved video: {video_path}")
                print(f"    {outcomes_str}")
            recording = False
            env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed,
                                render_mode=None)
            if hasattr(policy, "set_env"):
                policy.set_env(env)

        obs, _ = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        if executor:
            executor.reset()
        done = False
        terminated = False
        truncated = False
        frames = []
        chunk_targets = np.array([])
        chunk_idx = 0

        while not done:
            if executor:
                if executor.needs_query():
                    t0 = time.time()
                    state_t = torch.tensor(obs, dtype=torch.float32,
                                           device=DEVICE).unsqueeze(0)
                    pred = policy(state_t).cpu().numpy().flatten()
                    t_policy_total += time.time() - t0
                    n_policy_calls += 1
                    executor.set_chunk(pred)
                    chunk_targets = executor.get_all_targets()
                chunk_idx = executor.current_index()
                action = executor.get_action()
            else:
                t0 = time.time()
                state_t = torch.tensor(obs, dtype=torch.float32,
                                       device=DEVICE).unsqueeze(0)
                action = float(policy(state_t).cpu().numpy().flat[0])
                t_policy_total += time.time() - t0
                n_policy_calls += 1
                chunk_targets = np.array([action])
                chunk_idx = 0

            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            if recording and ep < video_episodes:
                t0 = time.time()
                frame = env.render()
                t_render_total += time.time() - t0
                if frame is not None:
                    frame = _draw_chunk_overlay(frame, chunk_targets, chunk_idx)
                    frames.append(frame)
            done = terminated or truncated

        episode_lengths.append(env.step_count)

        if (ep + 1) % max(1, num_episodes // 5) == 0 or ep == 0:
            elapsed = time.time() - t_eval_start
            avg_so_far = np.mean(episode_lengths)
            per_call = (t_policy_total / n_policy_calls * 1000) if n_policy_calls else 0
            print(f"    [eval] ep {ep+1}/{num_episodes} "
                  f"({elapsed:.1f}s elapsed, avg_len={avg_so_far:.0f}, "
                  f"policy: {t_policy_total:.1f}s/{n_policy_calls} calls "
                  f"= {per_call:.1f}ms/call, render: {t_render_total:.1f}s)")

        if recording and ep < video_episodes and frames:
            last_frame = frames[-1].copy()
            surface = pygame.surfarray.make_surface(
                np.transpose(last_frame, (1, 0, 2)))
            font = pygame.font.SysFont(None, 48)
            font_small = pygame.font.SysFont(None, 36)

            if truncated:
                text = font.render("TIMEOUT", True, (0, 255, 0))
                bg_color = (0, 100, 0)
            else:
                text = font.render("CRASHED", True, (255, 0, 0))
                bg_color = (100, 0, 0)

            pygame.draw.rect(surface, bg_color,
                             (10, 10, text.get_width() + 20,
                              text.get_height() + 20))
            surface.blit(text, (20, 20))

            ep_text = font_small.render(
                f"Episode {ep+1}/{video_episodes} - Steps: {len(frames)}",
                True, (255, 255, 255))
            pygame.draw.rect(surface, (0, 0, 0),
                             (10, 60, ep_text.get_width() + 20,
                              ep_text.get_height() + 20))
            surface.blit(ep_text, (20, 70))

            last_frame_annotated = np.transpose(
                pygame.surfarray.array3d(surface), (1, 0, 2))
            frames[-1] = last_frame_annotated
            for _ in range(60):
                frames.append(last_frame_annotated)
            all_frames.extend(frames)
            episode_outcomes.append(
                ("TIMEOUT" if truncated else "CRASHED", len(frames) - 60))

    env.close()
    elapsed = time.time() - t_eval_start
    per_call = (t_policy_total / n_policy_calls * 1000) if n_policy_calls else 0
    print(f"    [eval done] {num_episodes} eps in {elapsed:.1f}s | "
          f"policy: {t_policy_total:.1f}s ({n_policy_calls} calls, "
          f"{per_call:.1f}ms/call) | render: {t_render_total:.1f}s")

    if recording and all_frames:
        pygame.quit()
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        iio.imwrite(video_path, np.stack(all_frames), fps=30)
        outcomes_str = ", ".join(
            [f"Ep{i+1}: {steps}steps ({outcome})"
             for i, (outcome, steps) in enumerate(episode_outcomes)])
        print(f"  Saved video: {video_path}")
        print(f"    {outcomes_str}")

    return np.mean(episode_lengths), np.std(episode_lengths)


# ---------------------------------------------------------------------------
# DAgger helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout_and_relabel(policy, difficulty, num_episodes, pipe_speed, seed):
    """Roll out policy with chunk execution, relabel with deterministic expert.

    Uses DeterministicExpert which hovers at the midpoint when far from the
    pipe, commits to gap1 (upper gap) when close, and applies EMA smoothing
    — matching the original expert's trajectory style.
    """
    policy.eval()
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed)
    if hasattr(policy, "set_env"):
        policy.set_env(env)
    executor = ChunkExecutor()
    det_expert = DeterministicExpert()
    new_states, new_actions = [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        executor.reset()
        det_expert.reset()
        done = False
        ep_states, ep_expert_actions = [], []

        while not done:
            if executor.needs_query():
                state_t = torch.tensor(obs, dtype=torch.float32,
                                       device=DEVICE).unsqueeze(0)
                pred = policy(state_t).cpu().numpy().flatten()
                executor.set_chunk(pred)

            optimal_action = det_expert.act(obs)

            ep_states.append(obs.copy())
            ep_expert_actions.append(optimal_action)

            action = executor.get_action()
            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            done = terminated or truncated

        for i in range(len(ep_states) - ACTION_CHUNK + 1):
            new_states.append(ep_states[i])
            new_actions.append(ep_expert_actions[i:i + ACTION_CHUNK])

    env.close()
    if not new_states:
        return np.zeros((0, 4), dtype=np.float32), \
               np.zeros((0, ACTION_CHUNK), dtype=np.float32)
    return np.array(new_states, dtype=np.float32), \
           np.array(new_actions, dtype=np.float32)


def run_dagger(difficulty, initial_states, initial_actions, rounds,
               episodes_per_round, epochs, pipe_speed, seed,
               eval_episodes=100, verbose=False, video_dir=None):
    """Run DAgger training loop with automatic relabeling.

    If *video_dir* is given, saves a video for each round at
    ``video_dir/dagger_round_{rnd}.mp4``.
    """
    all_states = initial_states.copy()
    all_actions = initial_actions.copy()
    performance_means = []
    performance_stds = []

    for rnd in range(1, rounds + 1):
        print(f"  Round {rnd}/{rounds}: Training on {len(all_states)} "
              f"transitions...")

        policy = train_bc_policy(all_states, all_actions, epochs=epochs,
                                 verbose=verbose)

        video_path = None
        if video_dir is not None:
            video_path = f"{video_dir}/dagger_round_{rnd}.mp4"

        avg_len, std_len = evaluate_policy(
            policy, difficulty, num_episodes=eval_episodes,
            pipe_speed=pipe_speed, seed=seed + rnd * 1000,
            video_path=video_path, video_episodes=3)
        performance_means.append(avg_len)
        performance_stds.append(std_len)
        print(f"    Evaluation: {avg_len:.1f} ± {std_len:.1f} avg length "
              f"({eval_episodes} episodes)")

        print(f"    Collecting {episodes_per_round} episodes with "
              f"auto-relabeling...", end=" ", flush=True)
        new_states, new_actions = rollout_and_relabel(
            policy, difficulty, episodes_per_round,
            pipe_speed, seed + rnd * 10000)
        print(f"Got {len(new_states)} new transitions")

        if len(new_states) > 0:
            all_states = np.concatenate([new_states], axis=0)
            all_actions = np.concatenate([new_actions], axis=0)

    return policy, performance_means, performance_stds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    SEEDS = [1,2000, 3000]
    NUM_SEEDS = len(SEEDS)
    EVAL_SEED_HARD = 500
    EVAL_SEED_EASY = 600

    # ================================================================
    # PART 1 + 2: Hard mode — all methods, multi-seed
    # ================================================================
    print("=" * 60)
    print("PART 1+2: Hard Mode — All Methods")
    print(f"  Running {NUM_SEEDS} seeds: {SEEDS}")
    print("=" * 60)
    print("Expert hovers at midpoint, then randomly picks a gap.")
    print("BC regression averages the two modes -> crashes.")
    print("Diffusion models the full distribution -> succeeds.")

    # Expert baseline (deterministic policy, run once)
    print("\nEvaluating expert baseline (with video)...")
    expert_hard = ExpertWrapper("hard")
    expert_hard_mean, expert_hard_std = evaluate_policy(
        expert_hard, "hard", num_episodes=10, seed=EVAL_SEED_HARD,
        use_chunks=False,
        video_path="plots/hard_mode/expert_hard.mp4", video_episodes=3)
    print(f"  Expert hard: {expert_hard_mean:.1f} ± "
          f"{expert_hard_std:.1f} avg steps")

    bc_hard_means = []
    gauss_det_means = []
    gauss_stoch_means = []
    diff_hard_means = []
    dagger_final_means = []
    dagger_all_round_means = []

    for si, seed in enumerate(SEEDS):
        print(f"\n{'─' * 60}")
        print(f"  SEED {si+1}/{NUM_SEEDS} (seed={seed})")
        print(f"{'─' * 60}")

        seed_dir = f"plots/hard_mode/seed_{seed}"
        os.makedirs(seed_dir, exist_ok=True)

        # --- Collect hard-mode expert data ---
        print(f"\n  Collecting hard-mode expert data (seed={seed})...")
        hard_states, hard_actions = collect_expert_data(
            "hard", num_episodes=500, seed=seed)
        print(f"    Collected {len(hard_states)} chunk transitions")


        # --- DAgger ---
        print("\n  Running DAgger...")
        dagger_policy, dagger_round_means, dagger_round_stds = run_dagger(
            difficulty="hard",
            initial_states=hard_states,
            initial_actions=hard_actions,
            rounds=2,
            episodes_per_round=30,
            epochs=80,
            pipe_speed=PIPE_SPEED,
            seed=seed + 5000,
            eval_episodes=50,
            verbose=True,
            video_dir=seed_dir,
        )
        dagger_final_means.append(dagger_round_means[-1])
        dagger_all_round_means.append(dagger_round_means)
        print(f"    DAgger final: {dagger_round_means[-1]:.1f}")

        dagger_eval_m, dagger_eval_s = evaluate_policy(
            dagger_policy, "hard", num_episodes=100,
            seed=EVAL_SEED_HARD,
            video_path=f"{seed_dir}/dagger_hard.mp4", video_episodes=3)
        print(f"    DAgger (video eval): {dagger_eval_m:.1f} ± "
              f"{dagger_eval_s:.1f}")


        # --- BC (MSE regression) ---
        print("\n  Training BC (MSE regression)...")
        bc_hard = train_bc_policy(
            hard_states, hard_actions, epochs=80, verbose=True)
        m, s = evaluate_policy(
            bc_hard, "hard", num_episodes=50, seed=EVAL_SEED_HARD,
            video_path=f"{seed_dir}/bc_hard.mp4", video_episodes=3)
        bc_hard_means.append(m)
        print(f"    BC: {m:.1f} ± {s:.1f}")

        # --- Diffusion ---
        print("\n  Training Diffusion policy...")
        diff_hard = train_diffusion_policy(
            hard_states, hard_actions, epochs=50,
            T=NUM_DIFFUSION_ITERS, batch_size=2048, verbose=True)
        m, s = evaluate_policy(
            diff_hard, "hard", num_episodes=50, seed=EVAL_SEED_HARD,
            video_path=f"{seed_dir}/diffusion_hard.mp4", video_episodes=3)
        diff_hard_means.append(m)
        print(f"    Diffusion: {m:.1f} ± {s:.1f}")

        # --- Gaussian BC (NLL, learned variance) ---
        print("\n  Training Gaussian BC (NLL)...")
        gauss_hard = train_gaussian_bc_policy(
            hard_states, hard_actions, epochs=80, verbose=True)

        gauss_det = GaussianWrapper(gauss_hard, stochastic=False)
        m, s = evaluate_policy(
            gauss_det, "hard", num_episodes=50, seed=EVAL_SEED_HARD,
            video_path=f"{seed_dir}/gauss_det_hard.mp4", video_episodes=3)
        gauss_det_means.append(m)
        print(f"    Gauss (det): {m:.1f} ± {s:.1f}")

        gauss_stoch = GaussianWrapper(gauss_hard, stochastic=True)
        m, s = evaluate_policy(
            gauss_stoch, "hard", num_episodes=50, seed=EVAL_SEED_HARD,
            video_path=f"{seed_dir}/gauss_stoch_hard.mp4", video_episodes=3)
        gauss_stoch_means.append(m)
        print(f"    Gauss (stoch): {m:.1f} ± {s:.1f}")

        if si == NUM_SEEDS - 1:
            torch.save(bc_hard.state_dict(), "models/bc_hard_chunk.pt")
            torch.save(diff_hard.state_dict(),
                       "models/diffusion_hard_chunk.pt")
            torch.save(gauss_hard.state_dict(), "models/gauss_hard_chunk.pt")
            torch.save(dagger_policy.state_dict(),
                       "models/dagger_hard_chunk.pt")

        # --- Per-seed comparison chart ---
        fig, ax = plt.subplots(figsize=(14, 6))
        methods = ['BC MSE\n(det)', 'Gauss NLL\n(det)', 'Gauss NLL\n(stoch)',
                   'Diffusion\n(DDPM)', 'DAgger\n(upper gap)']
        seed_vals = [bc_hard_means[-1], gauss_det_means[-1],
                     gauss_stoch_means[-1], diff_hard_means[-1],
                     dagger_final_means[-1]]
        colors_bar = ['C0', 'C3', 'C4', 'C1', 'C2']
        bars = ax.bar(methods, seed_vals, color=colors_bar, alpha=0.8,
                      edgecolor='black', linewidth=2)
        ax.set_ylabel('Avg Episode Length (hard mode)', fontsize=12)
        ax.set_title(f'Five Approaches — Seed {seed}',
                     fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1200)
        ax.axhline(1000, color='gray', linestyle=':', linewidth=1,
                   alpha=0.5, label='Max steps (1000)')
        for bar, val in zip(bars, seed_vals):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 15,
                    f'{val:.0f}', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{seed_dir}/five_way_comparison.png', dpi=150)
        plt.close()
        print(f"    Saved: {seed_dir}/five_way_comparison.png")

    # --- Aggregate across seeds ---
    bc_hard_mean = np.mean(bc_hard_means)
    bc_hard_std = np.std(bc_hard_means)
    gauss_det_mean = np.mean(gauss_det_means)
    gauss_det_std = np.std(gauss_det_means)
    gauss_stoch_mean = np.mean(gauss_stoch_means)
    gauss_stoch_std = np.std(gauss_stoch_means)
    diff_hard_mean = np.mean(diff_hard_means)
    diff_hard_std = np.std(diff_hard_means)
    dagger_hard_mean = np.mean(dagger_final_means)
    dagger_hard_std = np.std(dagger_final_means)

    print(f"\n{'=' * 60}")
    print(f"AGGREGATED HARD-MODE RESULTS ({NUM_SEEDS} seeds: {SEEDS})")
    print(f"{'=' * 60}")
    print(f"  BC MSE:        {bc_hard_mean:.1f} ± {bc_hard_std:.1f}  "
          f"(per-seed: {[f'{v:.1f}' for v in bc_hard_means]})")
    print(f"  Gauss (det):   {gauss_det_mean:.1f} ± {gauss_det_std:.1f}  "
          f"(per-seed: {[f'{v:.1f}' for v in gauss_det_means]})")
    print(f"  Gauss (stoch): {gauss_stoch_mean:.1f} ± {gauss_stoch_std:.1f}  "
          f"(per-seed: {[f'{v:.1f}' for v in gauss_stoch_means]})")
    print(f"  Diffusion:     {diff_hard_mean:.1f} ± {diff_hard_std:.1f}  "
          f"(per-seed: {[f'{v:.1f}' for v in diff_hard_means]})")
    print(f"  DAgger:        {dagger_hard_mean:.1f} ± {dagger_hard_std:.1f}  "
          f"(per-seed: {[f'{v:.1f}' for v in dagger_final_means]})")

    # --- DAgger learning curve (averaged across seeds) ---
    print("\nCreating DAgger learning curve...")
    dagger_round_arr = np.array(dagger_all_round_means)
    dagger_round_avg = dagger_round_arr.mean(axis=0)
    dagger_round_se = dagger_round_arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    rounds_arr = np.arange(1, len(dagger_round_avg) + 1)
    ax.errorbar(rounds_arr, dagger_round_avg, yerr=dagger_round_se,
                marker='o', linewidth=2, markersize=8, capsize=5,
                capthick=2,
                label=f'DAgger (upper gap, {NUM_SEEDS} seeds)', color='C2')
    ax.axhline(bc_hard_mean, color='C0', linestyle='--', linewidth=2,
               label=f'BC regression ({bc_hard_mean:.0f} ± '
                     f'{bc_hard_std:.0f})')
    ax.fill_between(rounds_arr, bc_hard_mean - bc_hard_std,
                    bc_hard_mean + bc_hard_std, color='C0', alpha=0.1)
    ax.axhline(1000, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('DAgger Round', fontsize=12)
    ax.set_ylabel('Avg Episode Length (hard mode)', fontsize=12)
    ax.set_title(f'DAgger Resolves Bimodal Ambiguity ({NUM_SEEDS} seeds)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/hard_mode/dagger_hard_curve.png', dpi=150)
    plt.close()
    print("  Saved: plots/hard_mode/dagger_hard_curve.png")

    # --- Five-way comparison bar chart (aggregated) ---
    print("\nCreating aggregated five-way comparison chart...")
    fig, ax = plt.subplots(figsize=(14, 6))
    methods = ['BC MSE\n(det)', 'Gauss NLL\n(det)', 'Gauss NLL\n(stoch)',
               'Diffusion\n(DDPM)', 'DAgger\n(upper gap)']
    values = [bc_hard_mean, gauss_det_mean, gauss_stoch_mean,
              diff_hard_mean, dagger_hard_mean]
    errors = [bc_hard_std, gauss_det_std, gauss_stoch_std,
              diff_hard_std, dagger_hard_std]
    colors = ['C0', 'C3', 'C4', 'C1', 'C2']
    bars = ax.bar(methods, values, yerr=errors, capsize=10,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=2,
                  error_kw={'linewidth': 2, 'ecolor': 'black'})
    ax.set_ylabel('Avg Episode Length (hard mode)', fontsize=12)
    ax.set_title(f'Five Approaches to Bimodal Expert Data ({NUM_SEEDS} seeds)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1200)
    ax.axhline(1000, color='gray', linestyle=':', linewidth=1, alpha=0.5,
               label='Max steps (1000)')
    for bar, val, err in zip(bars, values, errors):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + err + 15,
                f'{val:.0f}±{err:.0f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('plots/hard_mode/five_way_comparison.png', dpi=150)
    plt.close()
    print("  Saved: plots/hard_mode/five_way_comparison.png")

    # ================================================================
    # PART 3: Easy mode sanity check — BC & Diffusion both work
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 3: Easy Mode Sanity Check (One Gap)")
    print(f"  Running {NUM_SEEDS} seeds: {SEEDS}")
    print("=" * 60)

    bc_easy_means = []
    diff_easy_means = []

    for si, seed in enumerate(SEEDS):
        print(f"\n  Seed {si+1}/{NUM_SEEDS} (seed={seed + 2000})...")
        easy_states, easy_actions = collect_expert_data(
            "easy", num_episodes=200, seed=seed + 2000)
        print(f"    Collected {len(easy_states)} transitions")

        bc_easy = train_bc_policy(easy_states, easy_actions, epochs=60)
        m, s = evaluate_policy(bc_easy, "easy", num_episodes=50,
                               seed=EVAL_SEED_EASY)
        bc_easy_means.append(m)
        print(f"    BC on easy: {m:.1f} ± {s:.1f}")

        diff_easy = train_diffusion_policy(
            easy_states, easy_actions, epochs=150, T=NUM_DIFFUSION_ITERS)
        m, s = evaluate_policy(diff_easy, "easy", num_episodes=50,
                               seed=EVAL_SEED_EASY)
        diff_easy_means.append(m)
        print(f"    Diff on easy: {m:.1f} ± {s:.1f}")

    bc_easy_mean = np.mean(bc_easy_means)
    bc_easy_std = np.std(bc_easy_means)
    diff_easy_mean = np.mean(diff_easy_means)
    diff_easy_std = np.std(diff_easy_means)

    # Bar chart: BC vs Diffusion on easy and hard
    os.makedirs("plots/easy_mode", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(2)
    width = 0.35
    bc_means_plot = [bc_easy_mean, bc_hard_mean]
    bc_stds_plot = [bc_easy_std, bc_hard_std]
    diff_means_plot = [diff_easy_mean, diff_hard_mean]
    diff_stds_plot = [diff_easy_std, diff_hard_std]

    bars1 = ax.bar(x - width / 2, bc_means_plot, width, yerr=bc_stds_plot,
                   capsize=8, label='BC (regression)', color='C0',
                   alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width / 2, diff_means_plot, width, yerr=diff_stds_plot,
                   capsize=8, label='Diffusion', color='C1',
                   alpha=0.8, edgecolor='black')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 15,
                    f'{h:.0f}', ha='center', va='bottom', fontsize=11,
                    fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['Easy (1 gap)', 'Hard (2 gaps)'], fontsize=12)
    ax.set_ylabel('Avg Episode Length', fontsize=12)
    ax.set_title(f'BC Regression vs Diffusion Policy ({NUM_SEEDS} seeds)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('plots/easy_mode/bc_vs_diffusion.png', dpi=150)
    plt.close()
    print(f"\n  Saved: plots/easy_mode/bc_vs_diffusion.png")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print(f"SUMMARY ({NUM_SEEDS} seeds: {SEEDS})")
    print("=" * 60)

    print(f"\nAction chunks: predict {ACTION_CHUNK}, execute first "
          f"{EXECUTE_STEPS}")

    print(f"\nPART 1 — Hard mode (mean ± std across {NUM_SEEDS} seeds):")
    print(f"  BC MSE (det):       {bc_hard_mean:.1f} ± {bc_hard_std:.1f}  "
          f"<- mean averages modes, crashes")
    print(f"  Gauss NLL (det):    {gauss_det_mean:.1f} ± {gauss_det_std:.1f}  "
          f"<- mean still averages, crashes")
    print(f"  Gauss NLL (stoch):  {gauss_stoch_mean:.1f} ± "
          f"{gauss_stoch_std:.1f}  <- samples randomly, no coherence")
    print(f"  Diffusion:          {diff_hard_mean:.1f} ± {diff_hard_std:.1f}  "
          f"<- models full distribution")

    print(f"\nPART 2 — DAgger on hard mode:")
    print(f"  DAgger final:       {dagger_hard_mean:.1f} ± "
          f"{dagger_hard_std:.1f}  <- resolves ambiguity")

    print(f"\nPART 3 — Easy mode:")
    print(f"  BC  on easy: {bc_easy_mean:.1f} ± {bc_easy_std:.1f}")
    print(f"  Diff on easy: {diff_easy_mean:.1f} ± {diff_easy_std:.1f}")

    print("\nAggregated plots:")
    print("  - plots/hard_mode/five_way_comparison.png")
    print("  - plots/hard_mode/dagger_hard_curve.png")
    print("  - plots/easy_mode/bc_vs_diffusion.png")
    print(f"\nPer-seed outputs (plots/hard_mode/seed_*):")
    for seed in SEEDS:
        print(f"  seed_{seed}/: five_way_comparison.png, "
              "bc_hard.mp4, diffusion_hard.mp4, "
              "gauss_det_hard.mp4, gauss_stoch_hard.mp4, dagger_hard.mp4, "
              "dagger_round_*.mp4")
    print("\nExpert video: plots/hard_mode/expert_hard.mp4")
    print("=" * 60)


if __name__ == "__main__":
    main()
