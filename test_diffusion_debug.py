"""Debug: analyze expert data distribution and train diffusion longer."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collect_data import Expert
from flappy_bird_env import FlappyBirdEnv
from train_bc import BCPolicy
from train_diffusion import NoisePredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"Device: {DEVICE}")

NUM_DIFFUSION_ITERS = 100
NUM_INFERENCE_ITERS = 100


def collect_expert_data(difficulty, num_episodes, seed=0):
    env = FlappyBirdEnv(difficulty=difficulty)
    expert = Expert()
    all_states, all_actions = [], []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        expert.reset()
        done = False
        while not done:
            a = expert.act(obs, difficulty)
            all_states.append(obs.copy())
            all_actions.append([a])
            obs, _, terminated, truncated, _ = env.step(np.array([a]))
            done = terminated or truncated
    env.close()
    return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.float32)


@torch.no_grad()
def evaluate(policy_fn, difficulty, num_episodes=30, seed=9000):
    env = FlappyBirdEnv(difficulty=difficulty)
    lengths = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action = policy_fn(s).cpu().numpy()[0]
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        lengths.append(env.step_count)
    env.close()
    return np.mean(lengths), np.std(lengths)


@torch.no_grad()
def sample_diffusion(model, noise_scheduler, state):
    batch = state.size(0)
    device = state.device
    noisy_action = torch.randn(batch, 1, device=device)
    noise_scheduler.set_timesteps(NUM_INFERENCE_ITERS)
    for k in noise_scheduler.timesteps:
        noise_pred = model(
            noisy_action, state,
            torch.full((batch,), k, device=device, dtype=torch.long)
        )
        noisy_action = noise_scheduler.step(
            model_output=noise_pred, timestep=k, sample=noisy_action
        ).prev_sample
    return ((noisy_action + 1.0) / 2.0).clamp(0.0, 1.0)


def main():
    # Step 1: Analyze expert data
    print("=" * 60)
    print("STEP 1: Analyze expert action distribution")
    print("=" * 60)

    states, actions = collect_expert_data("hard", 100, seed=0)
    print(f"Total transitions: {len(states)}")

    # Bin by distance to pipe
    dists = states[:, 0]
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5)]
    print(f"\nAction stats by distance to pipe:")
    for lo, hi in bins:
        mask = (dists >= lo) & (dists < hi)
        a = actions[mask]
        if len(a) > 0:
            gap1 = states[mask, 1]
            gap2 = states[mask, 2]
            # Check if actions cluster near gap1 or gap2
            near_gap1 = np.abs(a.flatten() - gap1) < 0.05
            near_gap2 = np.abs(a.flatten() - gap2) < 0.05
            midpoint = (gap1 + gap2) / 2
            near_mid = np.abs(a.flatten() - midpoint) < 0.05
            print(f"  dist [{lo:.1f}, {hi:.1f}): {len(a)} samples, "
                  f"near_gap1={near_gap1.sum()}, near_gap2={near_gap2.sum()}, "
                  f"near_mid={near_mid.sum()}")

    # Plot expert action distribution by distance
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, (lo, hi) in enumerate(bins):
        ax = axes[idx // 2][idx % 2]
        mask = (dists >= lo) & (dists < hi)
        a = actions[mask].flatten()
        ax.hist(a, bins=40, alpha=0.7)
        ax.set_title(f"dist in [{lo:.1f}, {hi:.1f}), n={len(a)}")
        ax.set_xlabel("Target y (normalized)")
    plt.suptitle("Expert action distribution by distance to pipe", fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/expert_actions_by_distance.png", dpi=150)
    print("\nSaved: plots/expert_actions_by_distance.png")

    # Step 2: Train diffusion with longer training
    print("\n" + "=" * 60)
    print("STEP 2: Train diffusion (500 epochs) vs BC")
    print("=" * 60)

    # Normalize actions to [-1, 1]
    actions_norm = actions * 2.0 - 1.0

    loader = DataLoader(
        TensorDataset(torch.tensor(states), torch.tensor(actions_norm)),
        batch_size=256, shuffle=True)

    model = NoisePredictor().to(DEVICE)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=NUM_DIFFUSION_ITERS,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )
    ema = EMAModel(parameters=model.parameters(), power=0.75)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)

    # Evaluate at checkpoints
    eval_epochs = [100, 300, 500]
    results = {}

    for epoch in range(1, 501):
        total_loss = 0.0; n = 0
        for s, a in loader:
            s, a = s.to(DEVICE), a.to(DEVICE)
            noise = torch.randn_like(a)
            t = torch.randint(0, NUM_DIFFUSION_ITERS, (s.size(0),), device=DEVICE).long()
            noisy_a = noise_scheduler.add_noise(a, noise, t)
            pred = model(noisy_a, s, t)
            loss = nn.MSELoss()(pred, noise)
            opt.zero_grad(); loss.backward(); opt.step()
            ema.step(model.parameters())
            total_loss += loss.item() * s.size(0); n += s.size(0)

        if epoch in eval_epochs:
            avg_loss = total_loss / n
            # Temporarily copy EMA weights for eval
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            model.eval()
            mean, std = evaluate(lambda s: sample_diffusion(model, noise_scheduler, s), "hard")
            results[epoch] = (mean, std, avg_loss)
            print(f"  Epoch {epoch}: loss={avg_loss:.6f}, eval={mean:.1f} ± {std:.1f}")
            ema.restore(model.parameters())
            model.train()

    # Also train BC for comparison
    print("\nTraining BC (50 epochs)...")
    bc_loader = DataLoader(
        TensorDataset(torch.tensor(states), torch.tensor(actions)),
        batch_size=256, shuffle=True)
    bc = BCPolicy().to(DEVICE)
    bc_opt = torch.optim.Adam(bc.parameters(), lr=1e-3)
    for _ in range(50):
        for s, a in bc_loader:
            s, a = s.to(DEVICE), a.to(DEVICE)
            loss = nn.MSELoss()(bc(s), a)
            bc_opt.zero_grad(); loss.backward(); bc_opt.step()
    bc.eval()
    bc_mean, bc_std = evaluate(lambda s: bc(s), "hard")
    print(f"  BC: {bc_mean:.1f} ± {bc_std:.1f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Epoch':>6} | {'Loss':>10} | {'Diffusion':>18}")
    print("-" * 45)
    for ep in eval_epochs:
        if ep in results:
            m, s, l = results[ep]
            print(f"{ep:>6} | {l:>10.6f} | {m:>7.1f} ± {s:>6.1f}")
    print(f"{'BC':>6} | {'':>10} | {bc_mean:>7.1f} ± {bc_std:>6.1f}")


if __name__ == "__main__":
    main()
