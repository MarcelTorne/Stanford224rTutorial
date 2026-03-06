"""Debug script to understand why DAgger isn't working.

Strategy:
1. Collect expert data on FAST pipes (our target distribution)
2. Train BC on fast expert data (upper bound)
3. Train BC on slow expert data, test on fast (baseline)
4. Run DAgger starting from slow data, adapting to fast
5. Compare all approaches

This will show us if the problem is:
- Data quality (expert on fast should get ~1000)
- Relabeling strategy (DAgger should approach expert on fast)
- Distribution shift (BC on slow -> fast gap)
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from collect_data import Expert
from flappy_bird_env import FlappyBirdEnv
from train_bc import BCPolicy
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"Using device: {DEVICE}")


def collect_expert_data(difficulty, num_episodes, pipe_speed, seed=0):
    """Collect expert demonstrations."""
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed)
    expert = Expert()
    all_states, all_actions = [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        expert.reset()
        done = False
        while not done:
            thrust = expert.act(obs, difficulty)
            all_states.append(obs.copy())
            all_actions.append([thrust])
            obs, _, terminated, truncated, _ = env.step(np.array([thrust]))
            done = terminated or truncated

    env.close()
    return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.float32)


def train_bc_policy(states, actions, epochs=100, verbose=False):
    """Train BC policy."""
    s_tensor = torch.tensor(states, dtype=torch.float32)
    a_tensor = torch.tensor(actions, dtype=torch.float32)
    loader = DataLoader(TensorDataset(s_tensor, a_tensor),
                        batch_size=256, shuffle=True)

    policy = BCPolicy().to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
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

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss/n:.6f}")

    return policy


@torch.no_grad()
def evaluate_policy(policy, difficulty, num_episodes, pipe_speed, seed):
    """Evaluate policy."""
    policy.eval()
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed)
    episode_lengths = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False

        while not done:
            state_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action = policy(state_t).detach().cpu().numpy()[0]
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        episode_lengths.append(env.step_count)

    env.close()
    return np.mean(episode_lengths), np.std(episode_lengths)


def main():
    difficulty = "easy"
    slow_speed = 2.0
    fast_speed = 7.0
    num_expert_episodes = 200
    eval_episodes = 100

    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    print("=" * 60)
    print("DEBUGGING: Why is DAgger failing?")
    print("=" * 60)

    # 1. Expert on FAST pipes (upper bound)
    print(f"\n1. Collecting expert data on FAST pipes (speed={fast_speed})...")
    fast_states, fast_actions = collect_expert_data(
        difficulty, num_expert_episodes, fast_speed, seed=0
    )
    print(f"   Collected {len(fast_states)} transitions")

    # 2. Expert on SLOW pipes (our current approach)
    print(f"\n2. Collecting expert data on SLOW pipes (speed={slow_speed})...")
    slow_states, slow_actions = collect_expert_data(
        difficulty, num_expert_episodes, slow_speed, seed=1000
    )
    print(f"   Collected {len(slow_states)} transitions")

    # 3. Train BC on FAST expert data (upper bound)
    print("\n3. Training BC on FAST expert data...")
    bc_fast_expert = train_bc_policy(fast_states, fast_actions, epochs=100, verbose=True)
    bc_fast_mean, bc_fast_std = evaluate_policy(
        bc_fast_expert, difficulty, eval_episodes, fast_speed, seed=5000
    )
    print(f"   BC (fast expert) on fast pipes: {bc_fast_mean:.1f} ± {bc_fast_std:.1f}")

    # 4. Train BC on SLOW expert data, test on fast (baseline)
    print("\n4. Training BC on SLOW expert data...")
    bc_slow_expert = train_bc_policy(slow_states, slow_actions, epochs=100, verbose=True)
    bc_slow_on_fast_mean, bc_slow_on_fast_std = evaluate_policy(
        bc_slow_expert, difficulty, eval_episodes, fast_speed, seed=6000
    )
    print(f"   BC (slow expert) on fast pipes: {bc_slow_on_fast_mean:.1f} ± {bc_slow_on_fast_std:.1f}")

    # 5. Compare
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"UPPER BOUND (BC on fast expert):  {bc_fast_mean:.1f} ± {bc_fast_std:.1f}")
    print(f"BASELINE (BC on slow expert):     {bc_slow_on_fast_mean:.1f} ± {bc_slow_on_fast_std:.1f}")
    print(f"\nGap to close: {bc_fast_mean - bc_slow_on_fast_mean:.1f} steps")
    print("\nIf BC on fast expert gets ~1000, the problem is solvable.")
    print("If BC on slow expert is similar, distribution shift isn't the issue.")
    print("=" * 60)

    # Save models for further testing
    torch.save(bc_fast_expert.state_dict(), "models/bc_fast_expert.pt")
    torch.save(bc_slow_expert.state_dict(), "models/bc_slow_expert.pt")


if __name__ == "__main__":
    main()
