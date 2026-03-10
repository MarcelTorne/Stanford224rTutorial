"""Main pipeline: imitation learning on Flappy Bird with action chunks.

Part 1 -- Hard mode (two gaps): BC regression averages bimodal expert actions
and crashes; Diffusion policy models the full distribution and succeeds.

Part 2 -- Hard mode (DAgger): starting from the same bimodal data, DAgger
relabels with a deterministic expert (always upper gap), gradually resolving
the ambiguity so BC regression succeeds.

Part 3 -- Easy mode sanity check: both BC and Diffusion work on unimodal data.

Action chunking: policies predict ACTION_CHUNK future target positions at once.
During rollout, only the first EXECUTE_STEPS are executed before re-predicting
(receding horizon).

Usage:
    srun --qos=high --gres=gpu:1 --pty bash
    conda activate flow-dpo
    python main.py
"""

import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from networks import BCPolicy, GaussianBCPolicy, TemporalNoisePredictor, DDPMSchedule
from losses import bc_loss, gaussian_nll_loss, diffusion_loss
from expert import collect_expert_data
from dagger import run_dagger
from visualization import (
    ExpertWrapper, DiffusionWrapper, GaussianWrapper, evaluate_policy,
)

# ---------------------------------------------------------------------------
# Constants (match quick_demo_chunks_v2.py exactly)
# ---------------------------------------------------------------------------
BC_BATCH_SIZE = 2048
BC_EPOCHS = 100
BC_LR = 1e-5
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
# Training helpers
# ---------------------------------------------------------------------------

def train_bc_policy(states, actions, epochs=50, batch_size=256, lr=1e-3,
                    verbose=False, device=DEVICE):
    """Train BC policy that outputs ACTION_CHUNK actions."""
    s_tensor = torch.tensor(states, dtype=torch.float32)
    a_tensor = torch.tensor(actions, dtype=torch.float32)
    action_dim = a_tensor.shape[1]
    loader = DataLoader(TensorDataset(s_tensor, a_tensor),
                        batch_size=batch_size, shuffle=True)

    policy = BCPolicy(action_dim=action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            pred = policy(s_batch)
            loss = bc_loss(pred, a_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * s_batch.size(0)
            n += s_batch.size(0)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n:.6f}")

    return policy


def train_gaussian_bc_policy(states, actions, epochs=50, batch_size=256,
                              lr=1e-3, verbose=False, device=DEVICE):
    """Train Gaussian BC policy (mean + learned variance) with NLL loss."""
    s_tensor = torch.tensor(states, dtype=torch.float32)
    a_tensor = torch.tensor(actions, dtype=torch.float32)
    action_dim = a_tensor.shape[1]
    loader = DataLoader(TensorDataset(s_tensor, a_tensor),
                        batch_size=batch_size, shuffle=True)

    policy = GaussianBCPolicy(action_dim=action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            mean, log_var = policy(s_batch)
            loss = gaussian_nll_loss(mean, log_var, a_batch)
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
                           verbose=False, device=DEVICE):
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
    ).to(device)
    schedule = DDPMSchedule(T=T, beta_start=beta_start, beta_end=beta_end,
                            device=device, action_dim=action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            bsz = s_batch.size(0)

            t = torch.randint(0, T, (bsz,), device=device)
            noisy_a, noise = schedule.q_sample(a_batch, t)
            pred_noise = model(noisy_a, s_batch, t)
            loss = diffusion_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bsz
            n += bsz

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n:.6f}")

    return DiffusionWrapper(model, schedule)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = f"plots/{timestamp}"
    models_dir = f"models/{timestamp}"
    os.makedirs("data", exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Run timestamp: {timestamp}")
    print(f"  Plots  -> {plots_dir}/")
    print(f"  Models -> {models_dir}/")

    # ================================================================
    # PART 1: Hard mode -- BC regression vs Diffusion (bimodal demo)
    # ================================================================
    print("=" * 60)
    print("PART 1: BC vs Diffusion on Hard Mode (Two Gaps)")
    print("=" * 60)
    print("Expert hovers at midpoint, then randomly picks a gap.")
    print("BC regression averages the two modes -> crashes.")
    print("Diffusion models the full distribution -> succeeds.")

    # 1a. Collect expert data on hard mode
    print("\n1a. Collecting hard-mode expert data...")
    hard_states, hard_actions = collect_expert_data(
        "hard", num_episodes=500, action_chunk=ACTION_CHUNK, seed=1)
    print(f"    Collected {len(hard_states)} chunk transitions")

    # Expert baseline (step-by-step, no chunks)
    print("\n    Evaluating expert (with video)...")
    expert_hard = ExpertWrapper("hard")
    expert_hard_mean, expert_hard_std = evaluate_policy(
        expert_hard, "hard", num_episodes=10, seed=500,
        use_chunks=False,
        video_path=f"{plots_dir}/expert_hard.mp4", video_episodes=3)
    print(f"    Expert hard: {expert_hard_mean:.1f} +/- "
          f"{expert_hard_std:.1f} avg steps")

    # 1b. Train BC (regression) on hard data
    print("\n1b. Training BC (MSE regression) on hard data...")
    bc_hard = train_bc_policy(hard_states, hard_actions, epochs=BC_EPOCHS,
                              lr=BC_LR, verbose=True, batch_size=BC_BATCH_SIZE)

    print("    Evaluating BC on hard mode (50 episodes)...")
    bc_hard_mean, bc_hard_std = evaluate_policy(
        bc_hard, "hard", num_episodes=50, seed=500,
        video_path=f"{plots_dir}/bc_hard.mp4", video_episodes=3)
    print(f"    BC  (regression): {bc_hard_mean:.1f} +/- "
          f"{bc_hard_std:.1f} avg steps")

    # 1c. Train Diffusion on hard data
    print("\n1c. Training Diffusion policy on hard data...")
    diff_hard = train_diffusion_policy(
        hard_states, hard_actions, epochs=50, T=NUM_DIFFUSION_ITERS,
        batch_size=BC_BATCH_SIZE, verbose=True)

    diff_hard_mean, diff_hard_std = evaluate_policy(
        diff_hard, "hard", num_episodes=50, seed=500,
        video_path=f"{plots_dir}/diffusion_hard.mp4", video_episodes=3)
    print(f"    Diffusion:        {diff_hard_mean:.1f} +/- "
          f"{diff_hard_std:.1f} avg steps")

    # 1d. Train Gaussian BC (learned variance) on hard data
    print("\n1d. Training Gaussian BC (NLL, learned variance) on hard data...")
    gauss_hard = train_gaussian_bc_policy(
        hard_states, hard_actions, epochs=BC_EPOCHS, lr=BC_LR, verbose=True)

    print("    Evaluating Gaussian BC (deterministic = mean only)...")
    gauss_det = GaussianWrapper(gauss_hard, stochastic=False)
    gauss_det_mean, gauss_det_std = evaluate_policy(
        gauss_det, "hard", num_episodes=50, seed=500,
        video_path=f"{plots_dir}/gauss_det_hard.mp4", video_episodes=3)
    print(f"    Gauss (det):      {gauss_det_mean:.1f} +/- "
          f"{gauss_det_std:.1f} avg steps")

    print("    Evaluating Gaussian BC (stochastic = sample from N(mu, sigma))...")
    gauss_stoch = GaussianWrapper(gauss_hard, stochastic=True)
    gauss_stoch_mean, gauss_stoch_std = evaluate_policy(
        gauss_stoch, "hard", num_episodes=50, seed=500,
        video_path=f"{plots_dir}/gauss_stoch_hard.mp4", video_episodes=3)
    print(f"    Gauss (stoch):    {gauss_stoch_mean:.1f} +/- "
          f"{gauss_stoch_std:.1f} avg steps")

    # 1e. Save models
    torch.save(bc_hard.state_dict(), f"{models_dir}/bc_hard_chunk.pt")
    torch.save(diff_hard.state_dict(), f"{models_dir}/diffusion_hard_chunk.pt")
    torch.save(gauss_hard.state_dict(), f"{models_dir}/gauss_hard_chunk.pt")

    # ================================================================
    # PART 2: Hard mode -- DAgger resolves bimodal ambiguity
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 2: DAgger on Hard Mode (deterministic expert)")
    print("=" * 60)
    print("DAgger relabels with an expert that always picks the upper gap.")
    print("Over rounds, the unimodal relabeled data resolves the ambiguity.\n")

    # 2a. Run DAgger starting from the pretrained BC policy
    print("2a. Running DAgger on hard mode (starting from pretrained BC, "
          "expert always -> upper gap)...")
    dagger_policy, dagger_means, dagger_stds = run_dagger(
        difficulty="hard",
        initial_states=hard_states,
        initial_actions=hard_actions,
        rounds=5,
        episodes_per_round=30,
        epochs=BC_EPOCHS,
        pipe_speed=PIPE_SPEED,
        seed=5000,
        action_chunk=ACTION_CHUNK,
        device=DEVICE,
        train_bc_fn=train_bc_policy,
        eval_episodes=50,
        verbose=True,
        initial_policy=bc_hard,
    )
    dagger_hard_mean = dagger_means[-1]
    dagger_hard_std = dagger_stds[-1]
    print(f"\n    DAgger final: {dagger_hard_mean:.1f} +/- "
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
               label=f'BC regression ({bc_hard_mean:.0f} +/- '
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
    plt.savefig(f'{plots_dir}/dagger_hard_curve.png', dpi=150)
    print(f"  Saved: {plots_dir}/dagger_hard_curve.png")

    # 2c. Five-way comparison bar chart
    print("\n2c. Creating five-way comparison chart...")
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
    ax.set_title('Five Approaches to Bimodal Expert Data',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1200)
    ax.axhline(1000, color='gray', linestyle=':', linewidth=1, alpha=0.5,
               label='Max steps (1000)')
    for bar, val, err in zip(bars, values, errors):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + err + 15,
                f'{val:.0f}+/-{err:.0f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/five_way_comparison.png', dpi=150)
    print(f"  Saved: {plots_dir}/five_way_comparison.png")

    # 2d. Evaluate DAgger (with video) and save model
    torch.save(dagger_policy.state_dict(), f"{models_dir}/dagger_hard_chunk.pt")
    print("\n2d. Evaluating DAgger (with video)...")
    dagger_eval_mean, dagger_eval_std = evaluate_policy(
        dagger_policy, "hard", num_episodes=100, seed=500,
        video_path=f"{plots_dir}/dagger_hard.mp4", video_episodes=3)
    print(f"    DAgger: {dagger_eval_mean:.1f} +/- "
          f"{dagger_eval_std:.1f} avg steps")

    # ================================================================
    # PART 3: Easy mode sanity check -- BC & Diffusion both work
    # ================================================================
    print("\n" + "=" * 60)
    print("PART 3: Easy Mode Sanity Check (One Gap)")
    print("=" * 60)

    print("\n3a. Collecting easy-mode expert data (sanity check)...")
    easy_states, easy_actions = collect_expert_data(
        "easy", num_episodes=500, action_chunk=ACTION_CHUNK, seed=2000)
    print(f"    Collected {len(easy_states)} transitions")

    print("    Training Diffusion on easy data...")
    diff_easy = train_diffusion_policy(
        easy_states, easy_actions, epochs=50, T=NUM_DIFFUSION_ITERS,
        batch_size=BC_BATCH_SIZE, verbose=True)
    diff_easy_mean, diff_easy_std = evaluate_policy(
        diff_easy, "easy", num_episodes=50, seed=600,
        video_path=f"{plots_dir}/diffusion_easy.mp4", video_episodes=3)
    print(f"    Diff on easy: {diff_easy_mean:.1f} +/- "
          f"{diff_easy_std:.1f} avg steps")

    print("    Training BC on easy data...")
    bc_easy = train_bc_policy(easy_states, easy_actions, epochs=BC_EPOCHS,
                              lr=BC_LR, batch_size=BC_BATCH_SIZE)
    bc_easy_mean, bc_easy_std = evaluate_policy(
        bc_easy, "easy", num_episodes=50, seed=600,
        video_path=f"{plots_dir}/bc_easy.mp4", video_episodes=3)
    print(f"    BC  on easy: {bc_easy_mean:.1f} +/- "
          f"{bc_easy_std:.1f} avg steps")

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
    plt.savefig(f'{plots_dir}/bc_vs_diffusion.png', dpi=150)
    print(f"\n  Saved: {plots_dir}/bc_vs_diffusion.png")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nAction chunks: predict {ACTION_CHUNK}, execute first "
          f"{EXECUTE_STEPS}")

    print("\nPART 1 -- Regression variants on hard mode:")
    print(f"  BC MSE (det):       {bc_hard_mean:.1f} +/- {bc_hard_std:.1f}  "
          f"<- mean averages modes, crashes")
    print(f"  Gauss NLL (det):    {gauss_det_mean:.1f} +/- {gauss_det_std:.1f}  "
          f"<- mean still averages, crashes")
    print(f"  Gauss NLL (stoch):  {gauss_stoch_mean:.1f} +/- {gauss_stoch_std:.1f}  "
          f"<- samples randomly, no coherence")
    print(f"  Diffusion:          {diff_hard_mean:.1f} +/- {diff_hard_std:.1f}  "
          f"<- models full distribution")

    print(f"\nPART 2 -- DAgger on hard mode:")
    print(f"  DAgger final: {dagger_hard_mean:.1f} +/- "
          f"{dagger_hard_std:.1f}  <- resolves ambiguity")

    print(f"\nPART 3 -- Easy mode sanity check:")
    print(f"  BC  on easy: {bc_easy_mean:.1f} +/- {bc_easy_std:.1f}")
    print(f"  Diff on easy: {diff_easy_mean:.1f} +/- {diff_easy_std:.1f}")

    print(f"\nRun timestamp: {timestamp}")
    print(f"\nPlots saved to {plots_dir}/:")
    print("  - bc_vs_diffusion.png")
    print("  - dagger_hard_curve.png")
    print("  - five_way_comparison.png")
    print(f"\nVideos saved to {plots_dir}/:")
    print("  - expert_hard.mp4")
    print("  - bc_hard.mp4, diffusion_hard.mp4, dagger_hard.mp4")
    print("  - gauss_det_hard.mp4, gauss_stoch_hard.mp4")
    print(f"\nModels saved to {models_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
