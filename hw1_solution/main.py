"""Main pipeline: imitation learning on Flappy Bird with action chunks.

Part 1 -- Easy mode: BC regression works on unimodal expert data.

Part 2 -- Hard mode: BC regression averages bimodal expert actions and crashes.

Part 3 -- Hard mode: Flow Matching models the full distribution and succeeds.

Part 4 -- Hard mode (DAgger): starting from BC, DAgger relabels with a
deterministic expert (always upper gap), resolving the ambiguity so BC succeeds.

Action chunking: policies predict ACTION_CHUNK future target positions at once.
During rollout, only the first EXECUTE_STEPS are executed before re-predicting
(receding horizon).

Usage:
    python main.py                              # run full pipeline
    python main.py --method bc --env easy       # just BC on easy mode
    python main.py --method bc --env hard       # BC on hard mode (observe failure)
    python main.py --method flow_matching --env hard  # flow matching on hard
    python main.py --method dagger --env hard   # DAgger on hard mode
"""

import argparse
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from networks import (
    BCPolicy, GaussianBCPolicy, DiffusionPolicy, FlowMatchingPolicy,
)
from losses import bc_loss, gaussian_nll_loss, diffusion_loss, flow_matching_loss
from expert import collect_expert_data
from dagger import run_dagger
from visualization import (
    ExpertWrapper, DiffusionWrapper, FlowMatchingWrapper, GaussianWrapper,
    evaluate_policy,
)

# ---------------------------------------------------------------------------
# Constants
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

def train_policy(model, loss_fn, states, actions, epochs=50, batch_size=256,
                 lr=1e-3, log_every=10, verbose=False, device=DEVICE):
    """Generic training loop for any policy.

    Args:
        model: the network to train (parameters will be optimized).
        loss_fn: callable(model, s_batch, a_batch) -> scalar loss.
        states/actions: numpy arrays of training data.
        epochs, batch_size, lr: training hyperparameters.
        log_every: print loss every N epochs when verbose=True.
        verbose: whether to print training progress.
        device: torch device.

    Returns:
        The trained model.
    """
    s_tensor = torch.tensor(states, dtype=torch.float32)
    a_tensor = torch.tensor(actions, dtype=torch.float32)
    loader = DataLoader(TensorDataset(s_tensor, a_tensor),
                        batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            loss = loss_fn(model, s_batch, a_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * s_batch.size(0)
            n += s_batch.size(0)

        if verbose and (epoch + 1) % log_every == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n:.6f}")

    return model


def train_bc_policy(states, actions, epochs=50, batch_size=256, lr=1e-3,
                    verbose=False, device=DEVICE):
    """Train BC policy that outputs ACTION_CHUNK actions."""
    action_dim = np.array(actions).shape[1]
    policy = BCPolicy(action_dim=action_dim).to(device)
    return train_policy(policy, bc_loss, states, actions,
                        epochs=epochs, batch_size=batch_size, lr=lr,
                        log_every=10, verbose=verbose, device=device)


def train_gaussian_bc_policy(states, actions, epochs=50, batch_size=256,
                              lr=1e-3, verbose=False, device=DEVICE):
    """Train Gaussian BC policy (mean + learned variance) with NLL loss."""
    action_dim = np.array(actions).shape[1]
    policy = GaussianBCPolicy(action_dim=action_dim).to(device)
    return train_policy(policy, gaussian_nll_loss, states, actions,
                        epochs=epochs, batch_size=batch_size, lr=lr,
                        log_every=10, verbose=verbose, device=device)


def train_diffusion_policy(states, actions, epochs=100, batch_size=256,
                           lr=1e-4, T=100, beta_start=0.0001, beta_end=0.02,
                           verbose=False, device=DEVICE):
    """Train a DDPM diffusion policy outputting ACTION_CHUNK actions."""
    action_dim = np.array(actions).shape[1]
    state_dim = np.array(states).shape[1]

    policy = DiffusionPolicy(
        state_dim=state_dim, pred_horizon=action_dim, action_dim=1,
        T=T, beta_start=beta_start, beta_end=beta_end, device=device,
    ).to(device)
    train_policy(policy, diffusion_loss, states, actions,
                 epochs=epochs, batch_size=batch_size, lr=lr,
                 log_every=20, verbose=verbose, device=device)
    return DiffusionWrapper(policy.model, policy.schedule)


def train_flow_matching_policy(states, actions, epochs=100, batch_size=256,
                               lr=1e-4, num_steps=20, verbose=False,
                               device=DEVICE):
    """Train a Flow Matching policy (same U-Net architecture as diffusion)."""
    action_dim = np.array(actions).shape[1]
    state_dim = np.array(states).shape[1]

    policy = FlowMatchingPolicy(
        state_dim=state_dim, pred_horizon=action_dim, action_dim=1,
        num_steps=num_steps, device=device,
    ).to(device)
    train_policy(policy, flow_matching_loss, states, actions,
                 epochs=epochs, batch_size=batch_size, lr=lr,
                 log_every=20, verbose=verbose, device=device)
    return FlowMatchingWrapper(policy.model, policy.schedule)


# ---------------------------------------------------------------------------
# Individual run functions
# ---------------------------------------------------------------------------

def _collect_data(difficulty, plots_dir):
    """Collect expert data and evaluate the expert baseline."""
    print(f"\nCollecting {difficulty}-mode expert data...")
    states, actions = collect_expert_data(
        difficulty, num_episodes=500, action_chunk=ACTION_CHUNK,
        seed=1 if difficulty == "hard" else 2000)
    print(f"    Collected {len(states)} chunk transitions")

    print(f"    Evaluating expert on {difficulty} (with video)...")
    expert = ExpertWrapper(difficulty)
    expert_mean, expert_std = evaluate_policy(
        expert, difficulty, num_episodes=10, seed=500,
        use_chunks=False,
        video_path=f"{plots_dir}/expert_{difficulty}.mp4", video_episodes=3)
    print(f"    Expert {difficulty}: {expert_mean:.1f} +/- {expert_std:.1f}")
    return states, actions


def run_bc(difficulty, states, actions, plots_dir, models_dir):
    print(f"\nTraining BC (MSE regression) on {difficulty} data...")
    policy = train_bc_policy(states, actions, epochs=BC_EPOCHS,
                             lr=BC_LR, verbose=True, batch_size=BC_BATCH_SIZE)
    mean, std = evaluate_policy(
        policy, difficulty, num_episodes=50, seed=500,
        video_path=f"{plots_dir}/bc_{difficulty}.mp4", video_episodes=3)
    print(f"    BC on {difficulty}: {mean:.1f} +/- {std:.1f}")
    torch.save(policy.state_dict(), f"{models_dir}/bc_{difficulty}_chunk.pt")
    return policy, mean, std


def run_flow_matching(difficulty, states, actions, plots_dir, models_dir):
    print(f"\nTraining Flow Matching policy on {difficulty} data...")
    policy = train_flow_matching_policy(
        states, actions, epochs=50, num_steps=NUM_DIFFUSION_ITERS,
        batch_size=BC_BATCH_SIZE, verbose=True)
    mean, std = evaluate_policy(
        policy, difficulty, num_episodes=50, seed=500,
        video_path=f"{plots_dir}/flow_matching_{difficulty}.mp4",
        video_episodes=3)
    print(f"    Flow Matching on {difficulty}: {mean:.1f} +/- {std:.1f}")
    torch.save(policy.state_dict(),
               f"{models_dir}/flow_matching_{difficulty}_chunk.pt")
    return policy, mean, std


def run_dagger_method(difficulty, states, actions, bc_policy, plots_dir,
                      models_dir):
    """Run DAgger. Trains BC first if no bc_policy provided."""
    if bc_policy is None:
        print("\n    Training BC policy as DAgger prerequisite...")
        bc_policy = train_bc_policy(states, actions, epochs=BC_EPOCHS,
                                    lr=BC_LR, batch_size=BC_BATCH_SIZE)

    print(f"\nRunning DAgger on {difficulty} mode "
          "(deterministic expert -> upper gap)...")
    policy, means, stds = run_dagger(
        difficulty=difficulty,
        initial_states=states,
        initial_actions=actions,
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
        initial_policy=bc_policy,
    )
    final_mean, final_std = means[-1], stds[-1]
    print(f"    DAgger final on {difficulty}: {final_mean:.1f} +/- {final_std:.1f}")

    torch.save(policy.state_dict(), f"{models_dir}/dagger_{difficulty}_chunk.pt")

    eval_mean, eval_std = evaluate_policy(
        policy, difficulty, num_episodes=100, seed=500,
        video_path=f"{plots_dir}/dagger_{difficulty}.mp4", video_episodes=3)
    print(f"    DAgger eval: {eval_mean:.1f} +/- {eval_std:.1f}")

    return policy, means, stds, final_mean, final_std


def _plot_dagger_curve(dagger_means, dagger_stds, bc_mean, bc_std, plots_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    rounds_arr = np.arange(1, len(dagger_means) + 1)
    dagger_means_arr = np.array(dagger_means)
    dagger_stds_arr = np.array(dagger_stds)

    ax.errorbar(rounds_arr, dagger_means_arr, yerr=dagger_stds_arr,
                marker='o', linewidth=2, markersize=8, capsize=5,
                capthick=2, label='DAgger (always upper gap)', color='C2')
    ax.axhline(bc_mean, color='C0', linestyle='--', linewidth=2,
               label=f'BC regression ({bc_mean:.0f} +/- {bc_std:.0f})')
    ax.fill_between(rounds_arr, bc_mean - bc_std, bc_mean + bc_std,
                    color='C0', alpha=0.1)
    ax.axhline(1000, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('DAgger Round', fontsize=12)
    ax.set_ylabel('Avg Episode Length (hard mode)', fontsize=12)
    ax.set_title('DAgger Resolves Bimodal Ambiguity', fontsize=14,
                 fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/dagger_hard_curve.png', dpi=150)
    plt.close()
    print(f"  Saved: {plots_dir}/dagger_hard_curve.png")


def _plot_comparison(bc_mean, bc_std, fm_mean, fm_std,
                     dagger_mean, dagger_std, plots_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['BC MSE\n(regression)', 'Flow Matching\n(OT-CFM)',
               'DAgger\n(upper gap)']
    values = [bc_mean, fm_mean, dagger_mean]
    errors = [bc_std, fm_std, dagger_std]
    colors = ['C0', 'C5', 'C2']
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
                f'{val:.0f}+/-{err:.0f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/comparison_hard.png', dpi=150)
    plt.close()
    print(f"  Saved: {plots_dir}/comparison_hard.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Imitation learning on Flappy Bird with action chunks.")
    parser.add_argument("--method", type=str, default="all",
                        choices=["bc", "flow_matching", "dagger", "all"],
                        help="Which method to run (default: all)")
    parser.add_argument("--env", type=str, default="all",
                        choices=["easy", "hard", "all"],
                        help="Which environment difficulty (default: all)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = f"plots/{timestamp}"
    models_dir = f"models/{timestamp}"
    os.makedirs("data", exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Run timestamp: {timestamp}")
    print(f"  Plots  -> {plots_dir}/")
    print(f"  Models -> {models_dir}/")
    print(f"  Method -> {args.method}")
    print(f"  Env    -> {args.env}")

    run_all = args.method == "all"
    envs = ["easy", "hard"] if args.env == "all" else [args.env]
    methods = (["bc", "flow_matching", "dagger"]
               if run_all else [args.method])

    results = {}

    for difficulty in envs:
        states, actions = _collect_data(difficulty, plots_dir)

        bc_policy = None

        if "bc" in methods:
            policy, mean, std = run_bc(difficulty, states, actions,
                                       plots_dir, models_dir)
            results[("bc", difficulty)] = (mean, std)
            bc_policy = policy

        if "flow_matching" in methods:
            _, mean, std = run_flow_matching(difficulty, states, actions,
                                             plots_dir, models_dir)
            results[("flow_matching", difficulty)] = (mean, std)

        if "dagger" in methods:
            _, dag_means, dag_stds, final_mean, final_std = run_dagger_method(
                difficulty, states, actions, bc_policy, plots_dir, models_dir)
            results[("dagger", difficulty)] = (final_mean, final_std)

            if difficulty == "hard":
                bc_hard = results.get(("bc", "hard"), (None, None))
                if bc_hard[0] is not None:
                    _plot_dagger_curve(dag_means, dag_stds,
                                       bc_hard[0], bc_hard[1], plots_dir)

    # ----- Comparison plot -----
    hard_keys = [("bc", "hard"), ("flow_matching", "hard"), ("dagger", "hard")]
    if all(k in results for k in hard_keys):
        print("\nCreating comparison chart...")
        _plot_comparison(
            *results[("bc", "hard")],
            *results[("flow_matching", "hard")],
            *results[("dagger", "hard")],
            plots_dir)

    # ----- Summary -----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nAction chunks: predict {ACTION_CHUNK}, execute first "
          f"{EXECUTE_STEPS}")
    for (method, difficulty), (mean, std) in sorted(results.items()):
        print(f"  {method:20s} [{difficulty:4s}]: {mean:.1f} +/- {std:.1f}")
    print(f"\nRun timestamp: {timestamp}")
    print(f"Plots saved to {plots_dir}/")
    print(f"Models saved to {models_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
