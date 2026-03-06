"""Run BC, Diffusion, and DAgger experiments across seeds and generate comparison plots.

Usage:
    python run_experiments.py                                              # full run
    python run_experiments.py --dagger_rounds 3 --dagger_epochs 20         # quick test
    python run_experiments.py --render_gifs                                # also save policy MP4s
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio.v3 as iio

from collect_data import Expert
from evaluate import load_policy, get_action
from flappy_bird_env import FlappyBirdEnv
from train_dagger import train_policy, rollout_and_label


def evaluate_episode_lengths(model_path, policy_type, difficulty, num_episodes, seed):
    """Run a trained policy and return per-episode step counts."""
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    policy = load_policy(model_path, policy_type, device)
    env = FlappyBirdEnv(difficulty=difficulty)

    episode_lengths = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False

        while not done:
            action = get_action(policy, obs, policy_type, device)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        episode_lengths.append(env.step_count)

    env.close()
    return episode_lengths


@torch.no_grad()
def evaluate_bc_policy_lengths(policy, device, difficulty, num_episodes, seed):
    """Evaluate an in-memory BC policy and return per-episode step counts."""
    policy.eval()
    env = FlappyBirdEnv(difficulty=difficulty)
    episode_lengths = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False

        while not done:
            state_t = torch.tensor(obs, dtype=torch.float32,
                                   device=device).unsqueeze(0)
            action = policy(state_t).cpu().numpy()[0]
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        episode_lengths.append(env.step_count)

    env.close()
    return episode_lengths


def run_dagger_seeded(difficulty, initial_data, rounds, episodes_per_round,
                      epochs, seed, save_path=None):
    """Run one full DAgger loop and return per-round average episode lengths."""
    data = np.load(initial_data)
    all_states = data["states"].copy()
    all_actions = data["actions"].copy()

    expert = Expert()
    avg_lengths = []

    for rnd in range(1, rounds + 1):
        policy, device = train_policy(all_states, all_actions, epochs=epochs)

        rollout_seed = seed + rnd * 10000
        new_states, new_actions, ep_lengths = rollout_and_label(
            policy, device, expert, difficulty,
            episodes_per_round, rollout_seed
        )

        avg_lengths.append(np.mean(ep_lengths))

        all_states = np.concatenate([all_states, new_states], axis=0)
        all_actions = np.concatenate([all_actions, new_actions], axis=0)

    # Save final model if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(policy.state_dict(), save_path)

    return avg_lengths


def run_dagger_reduced(difficulty, initial_data, max_transitions, rounds,
                       episodes_per_round, epochs, seed):
    """Run DAgger starting from a subset of the initial expert data."""
    data = np.load(initial_data)
    all_states = data["states"][:max_transitions].copy()
    all_actions = data["actions"][:max_transitions].copy()

    expert = Expert()
    avg_lengths = []

    for rnd in range(1, rounds + 1):
        policy, device = train_policy(all_states, all_actions, epochs=epochs)

        rollout_seed = seed + rnd * 10000
        new_states, new_actions, ep_lengths = rollout_and_label(
            policy, device, expert, difficulty,
            episodes_per_round, rollout_seed
        )

        avg_lengths.append(np.mean(ep_lengths))

        all_states = np.concatenate([all_states, new_states], axis=0)
        all_actions = np.concatenate([all_actions, new_actions], axis=0)

    return avg_lengths


def render_policy_video(model_path, policy_type, difficulty, output_path,
                        seed=42, max_frames=500):
    """Run one episode with rgb_array rendering and save as MP4."""
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    policy = load_policy(model_path, policy_type, device)
    env = FlappyBirdEnv(difficulty=difficulty, render_mode="rgb_array")

    obs, _ = env.reset(seed=seed)
    frames = []
    done = False

    while not done and len(frames) < max_frames:
        action = get_action(policy, obs, policy_type, device)
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        done = terminated or truncated

    env.close()

    if frames:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        iio.imwrite(output_path, np.stack(frames), fps=30)
        print(f"  Saved {len(frames)}-frame video to {output_path}")


def render_expert_video(difficulty, output_path, seed=42, max_frames=500):
    """Run one episode with the expert policy and save as MP4."""
    expert = Expert()
    env = FlappyBirdEnv(difficulty=difficulty, render_mode="rgb_array")

    obs, _ = env.reset(seed=seed)
    expert.reset()
    frames = []
    done = False

    while not done and len(frames) < max_frames:
        thrust = expert.act(obs, difficulty)
        obs, _, terminated, truncated, _ = env.step(np.array([thrust]))
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        done = terminated or truncated

    env.close()

    if frames:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        iio.imwrite(output_path, np.stack(frames), fps=30)
        print(f"  Saved {len(frames)}-frame video to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed experiments and plotting")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--dagger_rounds", type=int, default=10)
    parser.add_argument("--dagger_episodes", type=int, default=20)
    parser.add_argument("--dagger_epochs", type=int, default=100)
    parser.add_argument("--render_gifs", action="store_true",
                        help="Save MP4 visualizations of each policy")
    parser.add_argument("--reduced_demo_sizes", type=int, nargs="+",
                        default=[500, 5000],
                        help="Transition counts for reduced-demo DAgger experiment")
    args = parser.parse_args()

    seeds = args.seeds
    difficulties = ["easy", "hard"]

    # ------------------------------------------------------------------
    # 1. Evaluate BC
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Evaluating BC policies")
    print("=" * 60)
    bc_results = {}
    for diff in difficulties:
        model_path = f"models/bc_{diff}.pt"
        seed_means = []
        for s in seeds:
            lengths = evaluate_episode_lengths(
                model_path, "bc", diff, args.eval_episodes, seed=s * 1000
            )
            mean_len = np.mean(lengths)
            seed_means.append(mean_len)
            print(f"  BC {diff} seed={s}: avg episode length = {mean_len:.1f}")
        bc_results[diff] = seed_means

    # ------------------------------------------------------------------
    # 2. Evaluate Diffusion
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Evaluating Diffusion policies")
    print("=" * 60)
    diff_results = {}
    for diff in difficulties:
        model_path = f"models/diffusion_{diff}.pt"
        seed_means = []
        for s in seeds:
            lengths = evaluate_episode_lengths(
                model_path, "diffusion", diff, args.eval_episodes, seed=s * 1000
            )
            mean_len = np.mean(lengths)
            seed_means.append(mean_len)
            print(f"  Diffusion {diff} seed={s}: avg episode length = {mean_len:.1f}")
        diff_results[diff] = seed_means

    # ------------------------------------------------------------------
    # 3. Run DAgger (full data)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running DAgger experiments (full data)")
    print("=" * 60)
    dagger_curves = {diff: [] for diff in difficulties}
    dagger_final = {diff: [] for diff in difficulties}

    for diff in difficulties:
        initial_data = f"data/{diff}_expert.npz"
        for s in seeds:
            print(f"\n  DAgger {diff} seed={s}")
            # Save model from the last seed for video rendering
            save_path = f"models/dagger_{diff}.pt" if s == seeds[-1] else None
            curve = run_dagger_seeded(
                diff, initial_data, args.dagger_rounds,
                args.dagger_episodes, args.dagger_epochs, seed=s * 1000,
                save_path=save_path
            )
            dagger_curves[diff].append(curve)
            dagger_final[diff].append(curve[-1])
            print(f"    Final avg episode length: {curve[-1]:.1f}")

    # ------------------------------------------------------------------
    # 4. Reduced-demo DAgger experiment
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running reduced-demo DAgger experiment")
    print("=" * 60)
    reduced_results = {}

    for diff in difficulties:
        reduced_results[diff] = {}
        initial_data = f"data/{diff}_expert.npz"
        total_transitions = np.load(initial_data)["states"].shape[0]

        for size in args.reduced_demo_sizes:
            if size >= total_transitions:
                print(f"  Skipping size {size} for {diff} (>= {total_transitions} total)")
                continue
            print(f"\n  Reduced demos: {size} transitions, {diff} mode")
            reduced_results[diff][size] = {"dagger_curves": [], "bc_baselines": []}

            for s in seeds:
                # Train BC on reduced data and evaluate
                data = np.load(initial_data)
                policy, device = train_policy(
                    data["states"][:size], data["actions"][:size],
                    epochs=args.dagger_epochs
                )
                bc_lengths = evaluate_bc_policy_lengths(
                    policy, device, diff, args.eval_episodes, seed=s * 1000
                )
                bc_mean = np.mean(bc_lengths)
                reduced_results[diff][size]["bc_baselines"].append(bc_mean)
                print(f"    BC (reduced, {size}) seed={s}: {bc_mean:.1f}")

                # DAgger with reduced initial data
                curve = run_dagger_reduced(
                    diff, initial_data, size, args.dagger_rounds,
                    args.dagger_episodes, args.dagger_epochs, seed=s * 1000
                )
                reduced_results[diff][size]["dagger_curves"].append(curve)
                print(f"    DAgger (reduced, {size}) seed={s}: final={curve[-1]:.1f}")

    # ------------------------------------------------------------------
    # 5. Generate plots
    # ------------------------------------------------------------------
    os.makedirs("plots", exist_ok=True)

    # --- DAgger learning curves (2 subplots: easy, hard) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, diff in zip(axes, difficulties):
        curves = np.array(dagger_curves[diff])
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        rounds = np.arange(1, args.dagger_rounds + 1)

        ax.plot(rounds, mean_curve, marker="o", label="DAgger", color="C2")
        ax.fill_between(rounds, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.2, color="C2")

        bc_mean = np.mean(bc_results[diff])
        ax.axhline(bc_mean, linestyle="--", color="C0", label="BC")

        diff_mean = np.mean(diff_results[diff])
        ax.axhline(diff_mean, linestyle="--", color="C1", label="Diffusion")

        ax.set_xlabel("DAgger Round")
        ax.set_ylabel("Avg Episode Length")
        ax.set_title(f"DAgger Learning Curve — {diff}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("plots/dagger_learning_curves.png", dpi=150)
    print("\nSaved plots/dagger_learning_curves.png")

    # --- Final comparison bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(difficulties))
    width = 0.25

    for i, (method, results) in enumerate([
        ("BC", bc_results),
        ("Diffusion", diff_results),
        ("DAgger", dagger_final),
    ]):
        means = [np.mean(results[d]) for d in difficulties]
        stds = [np.std(results[d]) for d in difficulties]
        ax.bar(x + i * width, means, width, yerr=stds, label=method,
               capsize=4, color=f"C{i}")

    ax.set_xticks(x + width)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.set_ylabel("Avg Episode Length")
    ax.set_title("Method Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig("plots/comparison_bar.png", dpi=150)
    print("Saved plots/comparison_bar.png")

    # --- Reduced-demo DAgger plots (one per difficulty) ---
    for diff in difficulties:
        sizes = sorted(reduced_results.get(diff, {}).keys())
        if not sizes:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        rounds = np.arange(1, args.dagger_rounds + 1)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sizes)))

        for size, color in zip(sizes, colors):
            res = reduced_results[diff][size]
            curves = np.array(res["dagger_curves"])
            mean_curve = curves.mean(axis=0)
            std_curve = curves.std(axis=0)

            ax.plot(rounds, mean_curve, marker="o",
                    label=f"DAgger ({size} init trans.)", color=color)
            ax.fill_between(rounds, mean_curve - std_curve,
                            mean_curve + std_curve, alpha=0.15, color=color)

            # BC baseline trained on same reduced data
            bc_mean = np.mean(res["bc_baselines"])
            ax.axhline(bc_mean, linestyle=":", color=color, alpha=0.7,
                        label=f"BC ({size} init trans.)")

        # Full-data baselines
        ax.axhline(np.mean(bc_results[diff]), linestyle="--", color="C0",
                    label="BC (full data)")
        ax.axhline(np.mean(diff_results[diff]), linestyle="--", color="C1",
                    label="Diffusion (full data)")

        ax.set_xlabel("DAgger Round")
        ax.set_ylabel("Avg Episode Length")
        ax.set_title(f"DAgger with Reduced Demos — {diff}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = f"plots/dagger_reduced_demos_{diff}.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Saved {plot_path}")

    # ------------------------------------------------------------------
    # 6. Render videos (optional)
    # ------------------------------------------------------------------
    if args.render_gifs:
        print("\n" + "=" * 60)
        print("Rendering policy videos")
        print("=" * 60)

        # Expert videos
        for diff in difficulties:
            render_expert_video(diff, f"plots/expert_{diff}.mp4")

        # Trained policy videos
        video_configs = [
            ("models/bc_easy.pt", "bc", "easy"),
            ("models/bc_hard.pt", "bc", "hard"),
            ("models/diffusion_easy.pt", "diffusion", "easy"),
            ("models/diffusion_hard.pt", "diffusion", "hard"),
            ("models/dagger_easy.pt", "bc", "easy"),   # DAgger uses BCPolicy
            ("models/dagger_hard.pt", "bc", "hard"),
        ]
        for model_path, policy_type, diff in video_configs:
            if os.path.exists(model_path):
                name = os.path.splitext(os.path.basename(model_path))[0]
                output = f"plots/{name}.mp4"
                render_policy_video(model_path, policy_type, diff, output)
            else:
                print(f"  Skipping {model_path} (not found)")


if __name__ == "__main__":
    main()
