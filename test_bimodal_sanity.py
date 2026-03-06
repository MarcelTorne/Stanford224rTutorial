"""Sanity check: can our diffusion model learn a simple bimodal distribution?

Generate data from a Gaussian mixture (modes at -0.5 and +0.5) conditioned on
a binary state, and check if diffusion can sample from both modes cleanly.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_diffusion import NoisePredictor

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"Device: {DEVICE}")

NUM_DIFFUSION_ITERS = 100


def generate_bimodal_data(n=50000):
    """State is 4D (to match our architecture), action is bimodal in [-1, 1].

    For each sample, randomly pick mode 0 or mode 1.
    Mode 0: action ~ N(-0.5, 0.05)
    Mode 1: action ~ N(+0.5, 0.05)
    State encodes which pipe gap (mimicking our flappy bird setup).
    """
    states = np.random.randn(n, 4).astype(np.float32) * 0.1  # low-variance state
    modes = np.random.randint(0, 2, n)
    actions = np.where(modes == 0,
                       np.random.normal(-0.5, 0.05, n),
                       np.random.normal(0.5, 0.05, n)).astype(np.float32)
    return states, actions[:, None]


def train_and_test(states, actions, epochs=200):
    loader = DataLoader(
        TensorDataset(torch.tensor(states), torch.tensor(actions)),
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

    for epoch in range(epochs):
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
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} loss={total_loss/n:.6f}")

    ema.copy_to(model.parameters())
    model.eval()

    # Sample from the model
    with torch.no_grad():
        test_state = torch.tensor(states[:1], device=DEVICE).repeat(1000, 1)
        noisy_action = torch.randn(1000, 1, device=DEVICE)
        noise_scheduler.set_timesteps(NUM_DIFFUSION_ITERS)
        for k in noise_scheduler.timesteps:
            noise_pred = model(
                noisy_action, test_state,
                torch.full((1000,), k, device=DEVICE, dtype=torch.long)
            )
            noisy_action = noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=noisy_action
            ).prev_sample
        samples = noisy_action.cpu().numpy().flatten()

    return samples


# Test 1: Pure bimodal
print("=" * 50)
print("Test 1: Bimodal Gaussian mixture")
print("=" * 50)
states, actions = generate_bimodal_data(50000)
print(f"Data: {len(states)} samples, actions in [{actions.min():.2f}, {actions.max():.2f}]")
print(f"Training...")
samples = train_and_test(states, actions, epochs=200)

print(f"\nSampled 1000 actions:")
print(f"  Mean: {samples.mean():.3f}, Std: {samples.std():.3f}")
print(f"  Range: [{samples.min():.3f}, {samples.max():.3f}]")
# Check bimodality
left = samples[samples < 0]
right = samples[samples >= 0]
print(f"  Left mode (< 0): {len(left)} samples, mean={left.mean():.3f}" if len(left) > 0 else "  No left mode samples!")
print(f"  Right mode (>= 0): {len(right)} samples, mean={right.mean():.3f}" if len(right) > 0 else "  No right mode samples!")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(actions.flatten(), bins=50, alpha=0.7, label='Training data')
axes[0].set_title("Training data (bimodal)")
axes[1].hist(samples, bins=50, alpha=0.7, color='orange', label='Diffusion samples')
axes[1].set_title("Diffusion samples")
for ax in axes:
    ax.set_xlabel("Action value")
    ax.legend()
plt.tight_layout()
plt.savefig("plots/bimodal_sanity_check.png", dpi=150)
print("\nSaved: plots/bimodal_sanity_check.png")
