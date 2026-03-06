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
    python quick_demo_chunks.py
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
from collect_data import Expert, compute_action, compute_action_gravity
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

def train_bc_policy(states, actions, epochs=50, batch_size=256, lr=1e-3,
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


def train_diffusion_policy(states, actions, epochs=100, batch_size=256,
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

    The expert always targets gap1 (upper gap) in hard mode.
    Labels are action chunks: (s_t, [expert_a_t, ..., expert_a_{t+K-1}]).
    """
    policy.eval()
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed)
    if hasattr(policy, "set_env"):
        policy.set_env(env)
    executor = ChunkExecutor()
    new_states, new_actions = [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        executor.reset()
        done = False
        ep_states, ep_expert_actions = [], []

        while not done:
            if executor.needs_query():
                state_t = torch.tensor(obs, dtype=torch.float32,
                                       device=DEVICE).unsqueeze(0)
                pred = policy(state_t).cpu().numpy().flatten()
                executor.set_chunk(pred)

            bird_y = obs[3]
            target_y = obs[1]
            optimal_action = compute_action(bird_y, target_y)

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
               eval_episodes=100, verbose=False):
    """Run DAgger training loop with automatic relabeling."""
    all_states = initial_states.copy()
    all_actions = initial_actions.copy()
    performance_means = []
    performance_stds = []

    for rnd in range(1, rounds + 1):
        print(f"  Round {rnd}/{rounds}: Training on {len(all_states)} "
              f"transitions...")

        policy = train_bc_policy(all_states, all_actions, epochs=epochs,
                                 verbose=verbose)

        avg_len, std_len = evaluate_policy(
            policy, difficulty, num_episodes=eval_episodes,
            pipe_speed=pipe_speed, seed=seed + rnd * 1000)
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
            all_states = np.concatenate([all_states, new_states], axis=0)
            all_actions = np.concatenate([all_actions, new_actions], axis=0)

    return policy, performance_means, performance_stds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # ================================================================
    # PART 1: Hard mode — BC regression vs Diffusion (bimodal demo)
    # ================================================================
    print("=" * 60)
    print("PART 1: BC vs Diffusion on Hard Mode (Two Gaps)")
    print("=" * 60)
    print("Expert hovers at midpoint, then randomly picks a gap.")
    print("BC regression averages the two modes -> crashes.")
    print("Diffusion models the full distribution -> succeeds.")

    # 1a. Collect expert data on hard mode
    print("\n1a. Collecting hard-mode expert data...")
    hard_states, hard_actions = collect_expert_data("hard", num_episodes=100,
                                                    seed=0)
    print(f"    Collected {len(hard_states)} chunk transitions")

    # Expert baseline (step-by-step, no chunks)
    print("\n    Evaluating expert (with video)...")
    expert_hard = ExpertWrapper("hard")
    expert_hard_mean, expert_hard_std = evaluate_policy(
        expert_hard, "hard", num_episodes=200, seed=500,
        use_chunks=False,
        video_path="plots/expert_hard.mp4", video_episodes=3)
    print(f"    Expert hard: {expert_hard_mean:.1f} ± "
          f"{expert_hard_std:.1f} avg steps")

    # 1c. Train Diffusion on hard data
    print("\n1c. Training Diffusion policy on hard data...")
    diff_hard = train_diffusion_policy(hard_states, hard_actions, epochs=50,
                                       T=NUM_DIFFUSION_ITERS, verbose=True)

    diff_hard_mean, diff_hard_std = evaluate_policy(
        diff_hard, "hard", num_episodes=100, seed=500,
        video_path="plots/diffusion_hard.mp4", video_episodes=3)
    print(f"    Diffusion:        {diff_hard_mean:.1f} ± "
          f"{diff_hard_std:.1f} avg steps")

    # 1b. Train BC (regression) on hard data
    print("\n1b. Training BC (MSE regression) on hard data...")
    bc_hard = train_bc_policy(hard_states, hard_actions, epochs=80,
                              verbose=True)

    print("    Evaluating BC on hard mode (100 episodes)...")
    bc_hard_mean, bc_hard_std = evaluate_policy(
        bc_hard, "hard", num_episodes=100, seed=500,
        video_path="plots/bc_hard.mp4", video_episodes=3)
    print(f"    BC  (regression): {bc_hard_mean:.1f} ± "
          f"{bc_hard_std:.1f} avg steps")

    # 1e. Save models
    torch.save(bc_hard.state_dict(), "models/bc_hard_chunk.pt")
    torch.save(diff_hard.state_dict(), "models/diffusion_hard_chunk.pt")

    # ================================================================
    # PART 2: Hard mode — DAgger resolves bimodal ambiguity
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 2: DAgger on Hard Mode (deterministic expert)")
    print("=" * 60)
    print("DAgger relabels with an expert that always picks the upper gap.")
    print("Over rounds, the unimodal relabeled data resolves the ambiguity.\n")

    # 2a. Run DAgger starting from the SAME bimodal expert data
    print("2a. Running DAgger on hard mode (expert always -> upper gap)...")
    dagger_policy, dagger_means, dagger_stds = run_dagger(
        difficulty="hard",
        initial_states=hard_states,
        initial_actions=hard_actions,
        rounds=10,
        episodes_per_round=30,
        epochs=80,
        pipe_speed=PIPE_SPEED,
        seed=5000,
        eval_episodes=100,
        verbose=True,
    )
    dagger_hard_mean = dagger_means[-1]
    dagger_hard_std = dagger_stds[-1]
    print(f"\n    DAgger final: {dagger_hard_mean:.1f} ± "
          f"{dagger_hard_std:.1f}")

    # 2b. DAgger learning curve
    print("\n2b. Creating DAgger learning curve...")
    fig, ax = plt.subplots(figsize=(10, 6))
    rounds_arr = np.arange(1, len(dagger_means) + 1)
    dagger_means_arr = np.array(dagger_means)
    dagger_stds_arr = np.array(dagger_stds)

    ax.errorbar(rounds_arr, dagger_means_arr, yerr=dagger_stds_arr,
                marker='o', linewidth=2, markersize=8, capsize=5,
                capthick=2, label='DAgger (always upper gap)', color='C2')
    ax.axhline(bc_hard_mean, color='C0', linestyle='--', linewidth=2,
               label=f'BC regression ({bc_hard_mean:.0f} ± '
                     f'{bc_hard_std:.0f})')
    ax.fill_between(rounds_arr, bc_hard_mean - bc_hard_std,
                    bc_hard_mean + bc_hard_std, color='C0', alpha=0.1)
    ax.axhline(1000, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('DAgger Round', fontsize=12)
    ax.set_ylabel('Avg Episode Length (hard mode)', fontsize=12)
    ax.set_title('DAgger Resolves Bimodal Ambiguity', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/dagger_hard_curve.png', dpi=150)
    print("  Saved: plots/dagger_hard_curve.png")

    # 2c. Three-way comparison bar chart
    print("\n2c. Creating three-way comparison chart...")
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['BC\n(regression)', 'Diffusion\n(DDPM)', 'DAgger\n(upper gap)']
    values = [bc_hard_mean, diff_hard_mean, dagger_hard_mean]
    errors = [bc_hard_std, diff_hard_std, dagger_hard_std]
    colors = ['C0', 'C1', 'C2']
    bars = ax.bar(methods, values, yerr=errors, capsize=10,
                  color=colors, alpha=0.8, edgecolor='black', linewidth=2,
                  error_kw={'linewidth': 2, 'ecolor': 'black'})
    ax.set_ylabel('Avg Episode Length (hard mode)', fontsize=12)
    ax.set_title('Three Approaches to Bimodal Expert Data',
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
    plt.savefig('plots/three_way_comparison.png', dpi=150)
    print("  Saved: plots/three_way_comparison.png")

    # 2d. Evaluate DAgger (with video) and save model
    torch.save(dagger_policy.state_dict(), "models/dagger_hard_chunk.pt")
    print("\n2d. Evaluating DAgger (with video)...")
    dagger_eval_mean, dagger_eval_std = evaluate_policy(
        dagger_policy, "hard", num_episodes=100, seed=500,
        video_path="plots/dagger_hard.mp4", video_episodes=3)
    print(f"    DAgger: {dagger_eval_mean:.1f} ± "
          f"{dagger_eval_std:.1f} avg steps")

    # ================================================================
    # PART 3: Easy mode sanity check — BC & Diffusion both work
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 3: Easy Mode Sanity Check (One Gap)")
    print("=" * 60)

    print("\n3a. Collecting easy-mode expert data (sanity check)...")
    easy_states, easy_actions = collect_expert_data("easy", num_episodes=200,
                                                    seed=2000)
    print(f"    Collected {len(easy_states)} transitions")

    print("    Training BC on easy data...")
    bc_easy = train_bc_policy(easy_states, easy_actions, epochs=60)
    bc_easy_mean, bc_easy_std = evaluate_policy(bc_easy, "easy",
                                                num_episodes=100, seed=600)
    print(f"    BC  on easy: {bc_easy_mean:.1f} ± "
          f"{bc_easy_std:.1f} avg steps")

    print("    Training Diffusion on easy data...")
    diff_easy = train_diffusion_policy(easy_states, easy_actions,
                                       epochs=100, T=NUM_DIFFUSION_ITERS)
    diff_easy_mean, diff_easy_std = evaluate_policy(diff_easy, "easy",
                                                    num_episodes=100,
                                                    seed=600)
    print(f"    Diff on easy: {diff_easy_mean:.1f} ± "
          f"{diff_easy_std:.1f} avg steps")

    # 3b. Bar chart: BC vs Diffusion on easy and hard
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(2)
    width = 0.35
    bc_means = [bc_easy_mean, bc_hard_mean]
    bc_stds_plot = [bc_easy_std, bc_hard_std]
    diff_means = [diff_easy_mean, diff_hard_mean]
    diff_stds_plot = [diff_easy_std, diff_hard_std]

    bars1 = ax.bar(x - width / 2, bc_means, width, yerr=bc_stds_plot,
                   capsize=8, label='BC (regression)', color='C0',
                   alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width / 2, diff_means, width, yerr=diff_stds_plot,
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
    ax.set_title('BC Regression vs Diffusion Policy (Action Chunks)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('plots/bc_vs_diffusion.png', dpi=150)
    print("\n  Saved: plots/bc_vs_diffusion.png")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nAction chunks: predict {ACTION_CHUNK}, execute first "
          f"{EXECUTE_STEPS}")

    print("\nPART 1 — BC vs Diffusion on hard mode:")
    print(f"  BC  on hard: {bc_hard_mean:.1f} +/- {bc_hard_std:.1f}  "
          f"<- crashes (mode averaging)")
    print(f"  Diff on hard: {diff_hard_mean:.1f} +/- {diff_hard_std:.1f}  "
          f"<- models both modes")

    print(f"\nPART 2 — DAgger on hard mode:")
    print(f"  DAgger final: {dagger_hard_mean:.1f} +/- "
          f"{dagger_hard_std:.1f}  <- resolves ambiguity")

    print(f"\nPART 3 — Easy mode sanity check:")
    print(f"  BC  on easy: {bc_easy_mean:.1f} +/- {bc_easy_std:.1f}")
    print(f"  Diff on easy: {diff_easy_mean:.1f} +/- {diff_easy_std:.1f}")

    print("\nPlots saved:")
    print("  - plots/bc_vs_diffusion.png")
    print("  - plots/dagger_hard_curve.png")
    print("  - plots/three_way_comparison.png")
    print("\nVideos saved:")
    print("  - plots/expert_hard.mp4")
    print("  - plots/bc_hard.mp4, plots/diffusion_hard.mp4, "
          "plots/dagger_hard.mp4")
    print("=" * 60)


if __name__ == "__main__":
    main()
