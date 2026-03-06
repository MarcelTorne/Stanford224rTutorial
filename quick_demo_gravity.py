"""Quick demo: Train policies and show DAgger adaptation from slow to high gravity.

Optimized for CPU training in under 10 minutes.

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

from collect_data import Expert
from flappy_bird_env import FlappyBirdEnv
from train_bc import BCPolicy

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(f"Using device: {DEVICE}")


def collect_expert_data(difficulty, num_episodes, gravity=3.0, seed=0):
    """Collect expert demonstrations."""
    env = FlappyBirdEnv(difficulty=difficulty, gravity=gravity)
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


@torch.no_grad()
def evaluate_policy(policy, difficulty, num_episodes, gravity=3.0, seed=100):
    """Evaluate policy and return average episode length."""
    policy.eval()
    env = FlappyBirdEnv(difficulty=difficulty, gravity=gravity)
    episode_lengths = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False

        while not done:
            state_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action = policy(state_t).cpu().numpy()[0]
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        episode_lengths.append(env.step_count)

    env.close()
    return np.mean(episode_lengths), np.std(episode_lengths)


@torch.no_grad()
def rollout_and_relabel(policy, difficulty, num_episodes, gravity, seed):
    """Roll out policy and relabel ALL actions using EXPERT."""
    policy.eval()
    env = FlappyBirdEnv(difficulty=difficulty, gravity=gravity)
    expert = Expert()
    new_states, new_actions = [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        expert.reset()
        done = False

        while not done:
            state_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            policy_action = policy(state_t).detach().cpu().numpy()[0]

            optimal_action = expert.act(obs, difficulty)

            new_states.append(obs.copy())
            new_actions.append([optimal_action])

            obs, _, terminated, truncated, _ = env.step(policy_action)
            done = terminated or truncated

    env.close()
    return np.array(new_states, dtype=np.float32), np.array(new_actions, dtype=np.float32)


def run_dagger(difficulty, initial_states, initial_actions, rounds, episodes_per_round,
               epochs, gravity, seed, eval_episodes=100, verbose=False):
    """Run DAgger training loop with automatic relabeling."""
    all_states = initial_states.copy()
    all_actions = initial_actions.copy()
    performance_means = []
    performance_stds = []

    for rnd in range(1, rounds + 1):
        low_grav_pct = (len(initial_states) / len(all_states)) * 100 if rnd > 1 else 100
        high_grav_pct = 100 - low_grav_pct
        print(f"\n  ═══ Round {rnd}/{rounds} ═══")
        print(f"    Dataset: {len(all_states)} total transitions")
        print(f"      └─ Low gravity (0.5): {len(initial_states)} ({low_grav_pct:.1f}%)")
        if rnd > 1:
            print(f"      └─ High gravity (1.5): {len(all_states) - len(initial_states)} ({high_grav_pct:.1f}%)")
        print(f"    Training BC policy ({epochs} epochs)...")

        # Train on aggregated data
        policy = train_bc_policy(all_states, all_actions, epochs=epochs, verbose=verbose)

        # Evaluate with more episodes for better statistics
        print(f"    Evaluating on HIGH gravity (1.5) with {eval_episodes} episodes...")
        avg_len, std_len = evaluate_policy(policy, difficulty, num_episodes=eval_episodes,
                                           gravity=gravity, seed=seed + rnd * 1000)
        performance_means.append(avg_len)
        performance_stds.append(std_len)
        print(f"      → Performance: {avg_len:.1f} ± {std_len:.1f} avg steps")

        # Collect more data with AUTOMATIC relabeling
        print(f"    Collecting {episodes_per_round} episodes on HIGH gravity (1.5) + expert relabeling...")
        new_states, new_actions = rollout_and_relabel(
            policy, difficulty, episodes_per_round,
            gravity, seed + rnd * 10000
        )
        print(f"      → Collected {len(new_states)} new transitions from HIGH gravity")

        all_states = np.concatenate([all_states, new_states], axis=0)
        all_actions = np.concatenate([all_actions, new_actions], axis=0)

    return policy, performance_means, performance_stds


def render_video(policy, difficulty, gravity, output_path, seed=42, num_episodes=3):
    """Render multiple episodes with outcome overlays."""
    import pygame

    policy.eval()
    pygame.init()  # Initialize pygame once at the start
    env = FlappyBirdEnv(difficulty=difficulty, gravity=gravity, render_mode="rgb_array")
    all_frames = []
    episode_outcomes = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        frames = []
        done = False
        terminated = False
        truncated = False

        while not done and len(frames) < 1000:
            state_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action = policy(state_t).detach().cpu().numpy()[0]
            obs, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            if frame is not None:
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

    # Configuration
    difficulty = "easy"  # Focus on easy mode for clear demonstration
    low_gravity = 0.5   # Normal gravity
    high_gravity = 1.5  # 3x stronger gravity - much harder!

    print("=" * 60)
    print("QUICK DEMO: DAgger Gravity Adaptation")
    print("=" * 60)
    print(f"\nSETUP:")
    print(f"  • Low gravity (training): {low_gravity}")
    print(f"  • High gravity (test/adaptation): {high_gravity}")
    print(f"  • Goal: Show DAgger can adapt from low → high gravity")

    # Step 1: Collect expert data on SLOW pipes
    print(f"\n{'='*60}")
    print(f"STEP 1: Collect Expert Data on LOW Gravity")
    print(f"{'='*60}")
    print(f"Collecting 200 episodes with expert on gravity={low_gravity}...")
    states, actions = collect_expert_data(difficulty, num_episodes=200,
                                         gravity=low_gravity, seed=0)
    print(f"✓ Collected {len(states)} transitions from LOW gravity")

    # Step 2: Train initial BC policy on low gravity
    print(f"\n{'='*60}")
    print(f"STEP 2: Train BC Policy on LOW Gravity Data")
    print(f"{'='*60}")
    print(f"Training BC on {len(states)} transitions (60 epochs)...")
    bc_slow = train_bc_policy(states, actions, epochs=60, verbose=True)
    print(f"✓ BC policy trained")

    # Step 3: Evaluate BC on low gravity (should be good)
    print(f"\n{'='*60}")
    print(f"STEP 3: Evaluate BC Policy")
    print(f"{'='*60}")
    print(f"Evaluating BC on LOW gravity (gravity={low_gravity}, 100 episodes)...")
    bc_slow_on_slow, bc_slow_on_slow_std = evaluate_policy(bc_slow, difficulty, num_episodes=100,
                                                            gravity=low_gravity, seed=100)
    print(f"  ✓ BC on LOW gravity: {bc_slow_on_slow:.1f} ± {bc_slow_on_slow_std:.1f} avg steps")

    print(f"\nEvaluating BC on HIGH gravity (gravity={high_gravity}, 100 episodes)...")
    bc_slow_on_fast, bc_slow_on_fast_std = evaluate_policy(bc_slow, difficulty, num_episodes=100,
                                                            gravity=high_gravity, seed=200)
    print(f"  ✓ BC on HIGH gravity: {bc_slow_on_fast:.1f} ± {bc_slow_on_fast_std:.1f} avg steps")
    print(f"  → Distribution shift: {bc_slow_on_slow - bc_slow_on_fast:.1f} step drop!")

    # Step 5: Run DAgger to adapt to HIGH gravity
    print(f"\n{'='*60}")
    print(f"STEP 4: Run DAgger to Adapt to HIGH Gravity")
    print(f"{'='*60}")
    print(f"Starting dataset: {len(states)} LOW gravity transitions")
    print(f"Strategy: Collect 100 episodes/round on HIGH gravity + expert relabel")
    print(f"Goal: Overcome low-gravity bias and adapt to high gravity")
    dagger_policy, dagger_means, dagger_stds = run_dagger(
        difficulty=difficulty,
        initial_states=states,
        initial_actions=actions,
        rounds=15,  # More rounds
        episodes_per_round=100,  # WAY more episodes per round (was 20!)
        epochs=50,
        gravity=high_gravity,
        seed=1000,
        eval_episodes=100,
        verbose=True
    )

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"BC (trained on LOW gravity {low_gravity}):")
    print(f"  • On LOW gravity: {bc_slow_on_slow:.1f} ± {bc_slow_on_slow_std:.1f} steps")
    print(f"  • On HIGH gravity: {bc_slow_on_fast:.1f} ± {bc_slow_on_fast_std:.1f} steps")
    print(f"  • Distribution shift: -{bc_slow_on_slow - bc_slow_on_fast:.1f} steps")
    print(f"\nDAgger (adapted to HIGH gravity {high_gravity}):")
    print(f"  • Initial (Round 1): {dagger_means[0]:.1f} ± {dagger_stds[0]:.1f} steps")
    print(f"  • Final (Round {len(dagger_means)}): {dagger_means[-1]:.1f} ± {dagger_stds[-1]:.1f} steps")
    print(f"  • Improvement: +{dagger_means[-1] - dagger_means[0]:.1f} steps")
    recovery_pct = ((dagger_means[-1] - bc_slow_on_fast) / (bc_slow_on_slow - bc_slow_on_fast)) * 100
    print(f"  • Gap recovery: {recovery_pct:.1f}% of distribution shift")

    # Step 6: Create plots
    print(f"\n{'='*60}")
    print(f"STEP 5: Generate Visualizations")
    print(f"{'='*60}")

    # DAgger learning curve with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    rounds = np.arange(1, len(dagger_means) + 1)
    dagger_means_arr = np.array(dagger_means)
    dagger_stds_arr = np.array(dagger_stds)

    # Plot DAgger with error bars
    ax.errorbar(rounds, dagger_means_arr, yerr=dagger_stds_arr,
                marker='o', linewidth=2, markersize=8, capsize=5, capthick=2,
                label='DAgger (high gravity)', color='C0')

    # Plot BC baselines with shaded regions for std
    ax.axhline(bc_slow_on_slow, color='green', linestyle='--', linewidth=2,
               label=f'BC on low gravity ({bc_slow_on_slow:.0f} ± {bc_slow_on_slow_std:.0f})')
    ax.fill_between(rounds, bc_slow_on_slow - bc_slow_on_slow_std,
                     bc_slow_on_slow + bc_slow_on_slow_std,
                     color='green', alpha=0.1)

    ax.axhline(bc_slow_on_fast, color='red', linestyle='--', linewidth=2,
               label=f'BC on high gravity ({bc_slow_on_fast:.0f} ± {bc_slow_on_fast_std:.0f})')
    ax.fill_between(rounds, bc_slow_on_fast - bc_slow_on_fast_std,
                     bc_slow_on_fast + bc_slow_on_fast_std,
                     color='red', alpha=0.1)
    ax.set_xlabel('DAgger Round', fontsize=12)
    ax.set_ylabel('Avg Episode Length', fontsize=12)
    ax.set_title('DAgger Adaptation: Low → High Gravity', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/dagger_adaptation_curve.png', dpi=150)
    print("  Saved: plots/dagger_adaptation_curve.png")

    # Comparison bar chart with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['BC\n(low gravity)', 'BC\n(high gravity)', 'DAgger\n(high gravity)']
    values = [bc_slow_on_slow, bc_slow_on_fast, dagger_means[-1]]
    errors = [bc_slow_on_slow_std, bc_slow_on_fast_std, dagger_stds[-1]]
    colors = ['green', 'red', 'orange']

    bars = ax.bar(methods, values, yerr=errors, capsize=10,
                  color=colors, alpha=0.7, edgecolor='black', linewidth=2,
                  error_kw={'linewidth': 2, 'ecolor': 'black'})
    ax.set_ylabel('Avg Episode Length', fontsize=12)
    ax.set_title('Gravity Adaptation Performance (100 episodes each)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.3)

    # Add value labels on bars
    for bar, val, err in zip(bars, values, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err + 10,
                f'{val:.0f}±{err:.0f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('plots/speed_adaptation_comparison.png', dpi=150)
    print("  Saved: plots/speed_adaptation_comparison.png")

    # Step 7: Save models
    print("\n6. Saving models...")
    torch.save(bc_slow.state_dict(), "models/bc_slow.pt")
    torch.save(dagger_policy.state_dict(), "models/dagger_fast.pt")

    # Step 8: Render videos (3 episodes each)
    print("\n7. Rendering videos (3 episodes each)...")
    render_video(bc_slow, difficulty, low_gravity, "plots/bc_slow_on_slow.mp4", seed=42, num_episodes=3)
    render_video(bc_slow, difficulty, high_gravity, "plots/bc_slow_on_fast.mp4", seed=43, num_episodes=3)
    render_video(dagger_policy, difficulty, high_gravity, "plots/dagger_on_fast.mp4", seed=44, num_episodes=3)

    print("\n" + "=" * 60)
    print("SUMMARY (100 episodes per evaluation)")
    print("=" * 60)
    print(f"BC (trained on slow):  slow={bc_slow_on_slow:.1f}±{bc_slow_on_slow_std:.1f}  "
          f"fast={bc_slow_on_fast:.1f}±{bc_slow_on_fast_std:.1f}")
    print(f"DAgger (auto-relabel): fast={dagger_means[-1]:.1f}±{dagger_stds[-1]:.1f}")
    print(f"\nNote: DAgger uses EXPERT relabeling - policy rolls out on high gravity,")
    print(f"expert provides optimal labels for visited states, then policy retrains.")
    print(f"This allows adaptation from low gravity training to high gravity deployment!")

    if bc_slow_on_fast < bc_slow_on_slow:
        print(f"\n✓ BC performance dropped {bc_slow_on_slow - bc_slow_on_fast:.1f} steps on high gravity")

    if dagger_means[-1] > dagger_means[0]:
        print(f"✓ DAgger improved {dagger_means[-1] - dagger_means[0]:.1f} steps over {len(dagger_means)} rounds")

    print("\n📊 Plots saved:")
    print("  - plots/dagger_adaptation_curve.png")
    print("  - plots/speed_adaptation_comparison.png")
    print("\n🎥 Videos saved:")
    print("  - plots/bc_slow_on_slow.mp4")
    print("  - plots/bc_slow_on_fast.mp4")
    print("  - plots/dagger_on_fast.mp4")
    print("=" * 60)


if __name__ == "__main__":
    main()
