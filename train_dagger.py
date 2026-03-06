"""DAgger (Dataset Aggregation) for Flappy Bird behavior cloning.

Iteratively improves a BC policy by rolling out the current policy,
querying the expert for correct actions at visited states, and
retraining on the aggregated dataset.

Usage:
    python train_dagger.py --difficulty easy --rounds 10 --initial_data data/easy_expert.npz --output models/dagger_easy.pt
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from collect_data import Expert
from flappy_bird_env import FlappyBirdEnv
from train_bc import BCPolicy


def train_policy(states: np.ndarray, actions: np.ndarray, epochs: int = 100,
                 batch_size: int = 256, lr: float = 1e-3):
    """Train a fresh BCPolicy on the given dataset. Returns the trained policy."""
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    s_tensor = torch.tensor(states, dtype=torch.float32)
    a_tensor = torch.tensor(actions, dtype=torch.float32)
    loader = DataLoader(TensorDataset(s_tensor, a_tensor),
                        batch_size=batch_size, shuffle=True)

    policy = BCPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            pred = policy(s_batch)
            loss = loss_fn(pred, a_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * s_batch.size(0)
            n += s_batch.size(0)

    return policy, device


@torch.no_grad()
def rollout_and_label(policy, device, expert: Expert, difficulty: str,
                      num_episodes: int, seed: int):
    """Roll out policy, query expert at every state for relabeling.

    Returns new (states, actions) and the list of episode lengths.
    """
    policy.eval()
    env = FlappyBirdEnv(difficulty=difficulty)

    new_states = []
    new_actions = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        expert.reset()
        done = False

        while not done:
            state_t = torch.tensor(obs, dtype=torch.float32,
                                   device=device).unsqueeze(0)
            policy_action = policy(state_t).cpu().numpy()[0]

            expert_action = expert.act(obs, difficulty)
            new_states.append(obs.copy())
            new_actions.append([expert_action])

            obs, _, terminated, truncated, _ = env.step(policy_action)
            done = terminated or truncated

        episode_lengths.append(env.step_count)

    env.close()

    states = np.array(new_states, dtype=np.float32)
    actions = np.array(new_actions, dtype=np.float32)
    return states, actions, episode_lengths


def dagger(difficulty: str, rounds: int, episodes_per_round: int,
           initial_data: str, epochs: int, output: str, seed: int):
    # Load initial expert dataset
    data = np.load(initial_data)
    all_states = data["states"].copy()
    all_actions = data["actions"].copy()
    print(f"Initial dataset: {len(all_states)} transitions")

    expert = Expert()
    avg_lengths = []

    for rnd in range(1, rounds + 1):
        print(f"\n--- DAgger round {rnd}/{rounds} ---")

        # Train from scratch on aggregated dataset
        print(f"  Training on {len(all_states)} transitions for {epochs} epochs...")
        policy, device = train_policy(all_states, all_actions, epochs=epochs)

        # Roll out policy, collect expert-labeled data
        rollout_seed = seed + rnd * 10000
        new_states, new_actions, ep_lengths = rollout_and_label(
            policy, device, expert, difficulty,
            episodes_per_round, rollout_seed
        )

        avg_len = np.mean(ep_lengths)
        avg_lengths.append(avg_len)
        print(f"  Avg episode length: {avg_len:.1f}  "
              f"(min={np.min(ep_lengths)}, max={np.max(ep_lengths)})")

        # Aggregate
        all_states = np.concatenate([all_states, new_states], axis=0)
        all_actions = np.concatenate([all_actions, new_actions], axis=0)
        print(f"  Dataset size: {len(all_states)} transitions")

    # Save final model (retrain one last time on full dataset)
    print(f"\nFinal training on {len(all_states)} transitions...")
    policy, device = train_policy(all_states, all_actions, epochs=epochs)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(policy.state_dict(), output)
    print(f"Saved final DAgger policy to {output}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, rounds + 1), avg_lengths, marker="o")
    plt.xlabel("DAgger Round")
    plt.ylabel("Avg Episode Length")
    plt.title(f"DAgger — {difficulty} mode")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output.replace(".pt", "_episode_length.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="DAgger training for Flappy Bird")
    parser.add_argument("--difficulty", type=str, default="easy",
                        choices=["easy", "hard"])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--episodes_per_round", type=int, default=20)
    parser.add_argument("--initial_data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", type=str, default="models/dagger_easy.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dagger(args.difficulty, args.rounds, args.episodes_per_round,
           args.initial_data, args.epochs, args.output, args.seed)


if __name__ == "__main__":
    main()
