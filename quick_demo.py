"""Quick demo: three approaches to bimodal imitation learning.

Part 1 — Hard mode (two gaps): BC regression averages bimodal expert actions
and crashes; Diffusion policy models the full distribution and succeeds.

Part 2 — Hard mode (DAgger): starting from the same bimodal data, DAgger
relabels with a deterministic expert (always upper gap), gradually resolving
the ambiguity so BC regression succeeds.

Usage:
    python quick_demo.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collect_data import Expert, compute_action, compute_action_gravity, COMMIT_DIST
from flappy_bird_env import FlappyBirdEnv, SCREEN_W, SCREEN_H, BIRD_X
from train_bc import BCPolicy
from train_diffusion import NoisePredictor, DDPMSchedule

NUM_DIFFUSION_ITERS = 100
PIPE_SPEED = 3.0
ACTION_LOOKAHEAD = 80  # train on (s_t, a_{t+X}): anticipate X steps ahead
ADJUSTED_COMMIT_DIST = COMMIT_DIST + ACTION_LOOKAHEAD * PIPE_SPEED / SCREEN_W

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"Using device: {DEVICE}")
print(f"Action lookahead: {ACTION_LOOKAHEAD} steps, commit dist: {COMMIT_DIST:.3f} → {ADJUSTED_COMMIT_DIST:.3f}")


def collect_expert_data(difficulty, num_episodes, pipe_speed=PIPE_SPEED, seed=0):
    """Collect expert demonstrations using the same lookahead execution strategy.

    The expert is rolled out with the LookaheadExecutor (query every 20 steps,
    interpolate in between) so the visited states match what a learned policy
    will encounter.  At each step we also record the expert's per-step action
    (what it would say RIGHT NOW), then shift labels: (s_t, expert_a_{t+X}).
    """
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed)
    expert = Expert(commit_dist=ADJUSTED_COMMIT_DIST)
    executor = LookaheadExecutor()
    all_states, all_actions = [], []
    all_steps = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        expert.reset()
        executor.reset()
        done = False
        ep_states, ep_expert_actions = [], []
        while not done:
            expert_action = expert.act(obs, difficulty)
            ep_states.append(obs.copy())
            ep_expert_actions.append(expert_action)
            if executor.needs_query():
                executor.set_target(expert_action, bird_y=obs[3])
            action = executor.get_action()
            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            done = terminated or truncated
        all_steps.append(len(ep_states))
        for i in range(len(ep_states) - ACTION_LOOKAHEAD):
            all_states.append(ep_states[i])
            all_actions.append([ep_expert_actions[i + ACTION_LOOKAHEAD]])
    print(f"Average steps: {np.mean(all_steps):.1f}")
    print(f"Lookahead={ACTION_LOOKAHEAD}: {len(all_states)} shifted pairs from {num_episodes} episodes")
    env.close()
    return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.float32)


def train_bc_policy(states, actions, epochs=50, batch_size=256, lr=1e-3, verbose=False):
    """Train BC policy quickly."""
    s_tensor = torch.tensor(states, dtype=torch.float32)
    a_tensor = torch.tensor(actions, dtype=torch.float32)
    loader = DataLoader(TensorDataset(s_tensor, a_tensor),
                        batch_size=batch_size, shuffle=True)

    policy = BCPolicy().to(DEVICE)
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


class ExpertWrapper:
    """Wrap Expert so it has the same call interface as a learned policy.

    Stores a reference to the env so it can read bird_vel (oracle access).
    """

    def __init__(self, difficulty, env=None):
        self.expert = Expert(commit_dist=ADJUSTED_COMMIT_DIST)
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


class LookaheadExecutor:
    """Interpolates between lookahead policy predictions.

    Every ACTION_LOOKAHEAD steps the policy is queried for a new target.
    Between queries, actions are linearly interpolated from the previous
    target to the new one.
    """

    def __init__(self, lookahead=ACTION_LOOKAHEAD):
        self.lookahead = lookahead
        self.prev_target = None
        self.current_target = None
        self.step_in_window = 0

    def reset(self, bird_y=0.5):
        self.prev_target = None
        self.current_target = None
        self.step_in_window = 0

    def needs_query(self):
        return self.current_target is None or self.step_in_window >= self.lookahead

    def set_target(self, new_target, bird_y):
        self.prev_target = self.current_target if self.current_target is not None else bird_y
        self.current_target = new_target
        self.step_in_window = 0

    def get_action(self):
        alpha = (self.step_in_window + 1) / self.lookahead
        action = self.prev_target + (self.current_target - self.prev_target) * alpha
        self.step_in_window += 1
        return float(np.clip(action, 0.0, 1.0))


def train_diffusion_policy(states, actions, epochs=100, batch_size=256,
                           lr=1e-4, T=100, beta_start=0.0001, beta_end=0.02,
                           verbose=False):
    """Train a DDPM diffusion policy and return a DiffusionWrapper."""
    s_tensor = torch.tensor(states, dtype=torch.float32)
    a_tensor = torch.tensor(actions, dtype=torch.float32)
    loader = DataLoader(TensorDataset(s_tensor, a_tensor),
                        batch_size=batch_size, shuffle=True)

    model = NoisePredictor().to(DEVICE)
    schedule = DDPMSchedule(T=T, beta_start=beta_start, beta_end=beta_end,
                            device=DEVICE)
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


@torch.no_grad()
def evaluate_policy(policy, difficulty, num_episodes, pipe_speed=PIPE_SPEED,
                    seed=100, use_lookahead=True, video_path=None,
                    video_episodes=3):
    """Evaluate policy and return (mean, std) of episode lengths.

    If *video_path* is given, the first *video_episodes* episodes are recorded
    to an mp4 with prediction overlays — from the exact same rollouts that
    produce the statistics.
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
    executor = LookaheadExecutor() if use_lookahead else None
    episode_lengths = []
    all_frames = []
    episode_outcomes = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        if executor:
            executor.reset()
        done = False
        terminated = False
        truncated = False
        frames = []
        raw_target = None

        while not done:
            if executor:
                if executor.needs_query():
                    state_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    raw_target = float(policy(state_t).cpu().numpy().flat[0])
                    executor.set_target(raw_target, bird_y=obs[3])
                action = executor.get_action()
            else:
                state_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                action = float(policy(state_t).cpu().numpy().flat[0])
                raw_target = action
            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            if recording and ep < video_episodes:
                frame = env.render()
                if frame is not None:
                    frame = _draw_prediction_overlay(frame, raw_target, action)
                    frames.append(frame)
            done = terminated or truncated

        episode_lengths.append(env.step_count)

        if recording and ep < video_episodes and frames:
            last_frame = frames[-1].copy()
            surface = pygame.surfarray.make_surface(np.transpose(last_frame, (1, 0, 2)))
            font = pygame.font.SysFont(None, 48)
            font_small = pygame.font.SysFont(None, 36)

            if truncated:
                text = font.render("TIMEOUT", True, (0, 255, 0))
                bg_color = (0, 100, 0)
            else:
                text = font.render("CRASHED", True, (255, 0, 0))
                bg_color = (100, 0, 0)

            pygame.draw.rect(surface, bg_color,
                             (10, 10, text.get_width() + 20, text.get_height() + 20))
            surface.blit(text, (20, 20))

            ep_text = font_small.render(
                f"Episode {ep+1}/{video_episodes} - Steps: {len(frames)}",
                True, (255, 255, 255))
            pygame.draw.rect(surface, (0, 0, 0),
                             (10, 60, ep_text.get_width() + 20, ep_text.get_height() + 20))
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

    if recording:
        pygame.quit()
        if all_frames:
            os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
            iio.imwrite(video_path, np.stack(all_frames), fps=30)
            outcomes_str = ", ".join(
                [f"Ep{i+1}: {steps}steps ({outcome})"
                 for i, (outcome, steps) in enumerate(episode_outcomes)])
            print(f"  Saved video: {video_path}")
            print(f"    {outcomes_str}")

    return np.mean(episode_lengths), np.std(episode_lengths)


@torch.no_grad()
def rollout_and_relabel(policy, difficulty, num_episodes, pipe_speed, seed):
    """Roll out policy with interpolation and relabel with a deterministic expert.

    Easy mode: target the single gap.
    Hard mode: always target gap1 (upper gap), resolving bimodal ambiguity.

    Labels are shifted by ACTION_LOOKAHEAD: each pair is (s_t, expert_a_{t+X}).
    """
    policy.eval()
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed)
    if hasattr(policy, "set_env"):
        policy.set_env(env)
    executor = LookaheadExecutor()
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
                state_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                target = float(policy(state_t).detach().cpu().numpy().flat[0])
                executor.set_target(target, bird_y=obs[3])

            bird_y = obs[3]
            target_y = obs[1]
            optimal_action = compute_action(bird_y, target_y)

            ep_states.append(obs.copy())
            ep_expert_actions.append(optimal_action)

            action = executor.get_action()
            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            done = terminated or truncated

        for i in range(len(ep_states) - ACTION_LOOKAHEAD):
            new_states.append(ep_states[i])
            new_actions.append([ep_expert_actions[i + ACTION_LOOKAHEAD]])

    env.close()
    if not new_states:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)
    return np.array(new_states, dtype=np.float32), np.array(new_actions, dtype=np.float32)


def run_dagger(difficulty, initial_states, initial_actions, rounds, episodes_per_round,
               epochs, pipe_speed, seed, eval_episodes=100, verbose=False):
    """Run DAgger training loop with automatic relabeling."""
    all_states = initial_states.copy()
    all_actions = initial_actions.copy()
    performance_means = []
    performance_stds = []

    for rnd in range(1, rounds + 1):
        print(f"  Round {rnd}/{rounds}: Training on {len(all_states)} transitions...")

        policy = train_bc_policy(all_states, all_actions, epochs=epochs, verbose=verbose)

        avg_len, std_len = evaluate_policy(policy, difficulty, num_episodes=eval_episodes,
                                           pipe_speed=pipe_speed, seed=seed + rnd * 1000)
        performance_means.append(avg_len)
        performance_stds.append(std_len)
        print(f"    Evaluation: {avg_len:.1f} ± {std_len:.1f} avg length ({eval_episodes} episodes)")

        print(f"    Collecting {episodes_per_round} episodes with auto-relabeling...", end=" ", flush=True)
        new_states, new_actions = rollout_and_relabel(
            policy, difficulty, episodes_per_round,
            pipe_speed, seed + rnd * 10000
        )
        print(f"Got {len(new_states)} new transitions")

        if len(new_states) > 0:
            all_states = np.concatenate([all_states, new_states], axis=0)
            all_actions = np.concatenate([all_actions, new_actions], axis=0)

    return policy, performance_means, performance_stds


def _draw_prediction_overlay(frame, raw_target, interp_action):
    """Draw markers on a rendered frame showing the policy prediction.

    - Red diamond: raw policy prediction (the lookahead target)
    - Green line:  current interpolated action being executed
    """
    frame = frame.copy()
    h, w = frame.shape[:2]
    x = BIRD_X

    # Raw prediction — red diamond
    if raw_target is not None:
        ty = int(np.clip(raw_target * SCREEN_H, 0, SCREEN_H - 1))
        for dy in range(-5, 6):
            dx = 5 - abs(dy)
            for ddx in range(-dx, dx + 1):
                py, px = ty + dy, x + ddx + 30
                if 0 <= py < h and 0 <= px < w:
                    frame[py, px] = [255, 60, 60]

    # Interpolated action — green horizontal line
    ay = int(np.clip(interp_action * SCREEN_H, 0, SCREEN_H - 1))
    for dx in range(-20, 21):
        px = x + dx
        for thickness in range(-1, 2):
            py = ay + thickness
            if 0 <= py < h and 0 <= px < w:
                frame[py, px] = [60, 255, 60]

    return frame


@torch.no_grad()
def render_video(policy, difficulty, pipe_speed, output_path, seed=42,
                 num_episodes=3, use_lookahead=True):
    """Render multiple episodes with outcome overlays."""
    import pygame

    policy.eval()
    pygame.init()
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed,
                        render_mode="rgb_array")
    if hasattr(policy, "set_env"):
        policy.set_env(env)
    all_frames = []
    episode_outcomes = []

    executor = LookaheadExecutor() if use_lookahead else None

    for ep in range(num_episodes):
        torch.manual_seed(seed + ep)
        np.random.seed(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        if executor:
            executor.reset()
        frames = []
        done = False
        terminated = False
        truncated = False

        raw_target = None  # the policy's raw lookahead prediction

        while not done and len(frames) < 1000:
            if executor:
                if executor.needs_query():
                    state_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    raw_target = float(policy(state_t).detach().cpu().numpy().flat[0])
                    executor.set_target(raw_target, bird_y=obs[3])
                action = executor.get_action()
            else:
                state_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                action = float(policy(state_t).detach().cpu().numpy().flat[0])
                raw_target = action
            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            frame = env.render()
            if frame is not None:
                frame = _draw_prediction_overlay(frame, raw_target, action)
                frames.append(frame)
            done = terminated or truncated

        # Add outcome overlay to last frame of episode
        if frames:
            last_frame = frames[-1].copy()
            surface = pygame.surfarray.make_surface(np.transpose(last_frame, (1, 0, 2)))
            font = pygame.font.SysFont(None, 48)
            font_small = pygame.font.SysFont(None, 36)

            if truncated:
                text = font.render("TIMEOUT", True, (0, 255, 0))
                bg_color = (0, 100, 0)
            else:
                text = font.render("CRASHED", True, (255, 0, 0))
                bg_color = (100, 0, 0)

            # Draw background rectangle
            pygame.draw.rect(surface, bg_color, (10, 10, text.get_width() + 20, text.get_height() + 20))
            surface.blit(text, (20, 20))

            # Add episode number and steps
            ep_text = font_small.render(f"Episode {ep+1}/{num_episodes} - Steps: {len(frames)}",
                                       True, (255, 255, 255))
            pygame.draw.rect(surface, (0, 0, 0), (10, 60, ep_text.get_width() + 20, ep_text.get_height() + 20))
            surface.blit(ep_text, (20, 70))

            last_frame_annotated = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
            frames[-1] = last_frame_annotated

            # Hold last frame for 2 seconds
            for _ in range(60):
                frames.append(last_frame_annotated)

            all_frames.extend(frames)
            episode_outcomes.append(("TIMEOUT" if truncated else "CRASHED", len(frames) - 60))

    env.close()
    pygame.quit()

    if all_frames:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        iio.imwrite(output_path, np.stack(all_frames), fps=30)

        outcomes_str = ", ".join([f"Ep{i+1}: {steps}steps ({outcome})"
                                 for i, (outcome, steps) in enumerate(episode_outcomes)])
        print(f"  Saved video: {output_path}")
        print(f"    {outcomes_str}")


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
    hard_states, hard_actions = collect_expert_data("hard", num_episodes=100, seed=0)
    print(f"    Collected {len(hard_states)} transitions")

    print("\n    Evaluating expert (with video)...")
    expert_hard = ExpertWrapper("hard")
    expert_hard_mean, expert_hard_std = evaluate_policy(
        expert_hard, "hard", num_episodes=200, seed=500, use_lookahead=True,
        video_path="plots/expert_hard.mp4", video_episodes=5)
    print(f"    Expert hard: {expert_hard_mean:.1f} ± {expert_hard_std:.1f} avg steps")

    # 1c. Train Diffusion on hard data
    print("\n1c. Training Diffusion policy on hard data...")
    diff_hard = train_diffusion_policy(hard_states, hard_actions, epochs=150,
                                       T=NUM_DIFFUSION_ITERS, verbose=True)

    diff_hard_mean, diff_hard_std = evaluate_policy(
        diff_hard, "hard", num_episodes=100, seed=500,
        video_path="plots/diffusion_hard.mp4", video_episodes=3)
    print(f"    Diffusion:        {diff_hard_mean:.1f} ± {diff_hard_std:.1f} avg steps")


    # 1b. Train BC (regression) on hard data
    print("\n1b. Training BC (MSE regression) on hard data...")
    bc_hard = train_bc_policy(hard_states, hard_actions, epochs=80, verbose=True)

    print("    Evaluating BC on hard mode (100 episodes)...")
    bc_hard_mean, bc_hard_std = evaluate_policy(
        bc_hard, "hard", num_episodes=100, seed=500,
        video_path="plots/bc_hard.mp4", video_episodes=3)
    print(f"    BC  (regression): {bc_hard_mean:.1f} ± {bc_hard_std:.1f} avg steps")

    # 1e. Save models
    torch.save(bc_hard.state_dict(), "models/bc_hard.pt")
    torch.save(diff_hard.state_dict(), "models/diffusion_hard.pt")

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
    print(f"\n    DAgger final: {dagger_hard_mean:.1f} ± {dagger_hard_std:.1f}")

    # 2b. DAgger learning curve
    print("\n2b. Creating DAgger learning curve...")
    fig, ax = plt.subplots(figsize=(10, 6))
    rounds_arr = np.arange(1, len(dagger_means) + 1)
    dagger_means_arr = np.array(dagger_means)
    dagger_stds_arr = np.array(dagger_stds)

    ax.errorbar(rounds_arr, dagger_means_arr, yerr=dagger_stds_arr,
                marker='o', linewidth=2, markersize=8, capsize=5, capthick=2,
                label='DAgger (always upper gap)', color='C2')
    ax.axhline(bc_hard_mean, color='C0', linestyle='--', linewidth=2,
               label=f'BC regression ({bc_hard_mean:.0f} ± {bc_hard_std:.0f})')
    ax.fill_between(rounds_arr, bc_hard_mean - bc_hard_std,
                     bc_hard_mean + bc_hard_std, color='C0', alpha=0.1)
    ax.axhline(1000, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('DAgger Round', fontsize=12)
    ax.set_ylabel('Avg Episode Length (hard mode)', fontsize=12)
    ax.set_title('DAgger Resolves Bimodal Ambiguity', fontsize=14, fontweight='bold')
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
    torch.save(dagger_policy.state_dict(), "models/dagger_hard.pt")
    print("\n2d. Evaluating DAgger (with video)...")
    dagger_eval_mean, dagger_eval_std = evaluate_policy(
        dagger_policy, "hard", num_episodes=100, seed=500,
        video_path="plots/dagger_hard.mp4", video_episodes=3)
    print(f"    DAgger: {dagger_eval_mean:.1f} ± {dagger_eval_std:.1f} avg steps")

    # ================================================================
    # PART 3: Easy mode sanity check — BC & Diffusion both work
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 3: Easy Mode Sanity Check (One Gap)")
    print("=" * 60)

    print("\n3a. Collecting easy-mode expert data (sanity check)...")
    easy_states, easy_actions = collect_expert_data("easy", num_episodes=200, seed=2000)
    print(f"    Collected {len(easy_states)} transitions")

    print("    Training BC on easy data...")
    bc_easy = train_bc_policy(easy_states, easy_actions, epochs=60)
    bc_easy_mean, bc_easy_std = evaluate_policy(bc_easy, "easy", num_episodes=100, seed=600)
    print(f"    BC  on easy: {bc_easy_mean:.1f} ± {bc_easy_std:.1f} avg steps")

    print("    Training Diffusion on easy data...")
    diff_easy = train_diffusion_policy(easy_states, easy_actions, epochs=100, T=10)
    diff_easy_mean, diff_easy_std = evaluate_policy(diff_easy, "easy", num_episodes=100, seed=600)
    print(f"    Diff on easy: {diff_easy_mean:.1f} ± {diff_easy_std:.1f} avg steps")

    # 3b. Bar chart: BC vs Diffusion on easy and hard
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(2)
    width = 0.35
    bc_means = [bc_easy_mean, bc_hard_mean]
    bc_stds = [bc_easy_std, bc_hard_std]
    diff_means = [diff_easy_mean, diff_hard_mean]
    diff_stds = [diff_easy_std, diff_hard_std]

    bars1 = ax.bar(x - width / 2, bc_means, width, yerr=bc_stds, capsize=8,
                   label='BC (regression)', color='C0', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width / 2, diff_means, width, yerr=diff_stds, capsize=8,
                   label='Diffusion', color='C1', alpha=0.8, edgecolor='black')

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 15,
                    f'{h:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['Easy (1 gap)', 'Hard (2 gaps)'], fontsize=12)
    ax.set_ylabel('Avg Episode Length', fontsize=12)
    ax.set_title('BC Regression vs Diffusion Policy', fontsize=14, fontweight='bold')
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

    print("\nPART 1 — BC vs Diffusion on hard mode:")
    print(f"  BC  on hard: {bc_hard_mean:.1f} +/- {bc_hard_std:.1f}  <- crashes (mode averaging)")
    print(f"  Diff on hard: {diff_hard_mean:.1f} +/- {diff_hard_std:.1f}  <- models both modes")

    print(f"\nPART 2 — DAgger on hard mode (deterministic expert -> upper gap):")
    print(f"  DAgger final: {dagger_hard_mean:.1f} +/- {dagger_hard_std:.1f}  <- resolves ambiguity")

    print(f"\nPART 3 — Easy mode sanity check:")
    print(f"  BC  on easy: {bc_easy_mean:.1f} +/- {bc_easy_std:.1f}")
    print(f"  Diff on easy: {diff_easy_mean:.1f} +/- {diff_easy_std:.1f}")

    print("\nThree approaches to bimodal data:")
    print(f"  1. BC regression:  {bc_hard_mean:.1f}  (averages modes -> hovers -> crashes)")
    print(f"  2. Diffusion:      {diff_hard_mean:.1f}  (models full distribution -> picks a mode)")
    print(f"  3. DAgger:         {dagger_hard_mean:.1f}  (expert resolves ambiguity -> always upper)")

    print("\nPlots saved:")
    print("  - plots/bc_vs_diffusion.png")
    print("  - plots/dagger_hard_curve.png")
    print("  - plots/three_way_comparison.png")
    print("\nVideos saved:")
    print("  - plots/expert_hard.mp4")
    print("  - plots/bc_hard.mp4, plots/diffusion_hard.mp4, plots/dagger_hard.mp4")
    print("=" * 60)


if __name__ == "__main__":
    main()
