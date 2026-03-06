"""Evaluate a trained policy (BC or diffusion) in the Flappy Bird environment.

Usage:
    python evaluate.py --model models/bc_easy.pt --type bc --difficulty easy --episodes 50
    python evaluate.py --model models/diffusion_hard.pt --type diffusion --difficulty hard --episodes 50 --render
"""

import argparse

import numpy as np
import torch

from flappy_bird_env import FlappyBirdEnv
from train_bc import BCPolicy
from train_diffusion import NoisePredictor, DDPMSchedule


def load_policy(model_path: str, policy_type: str, device: torch.device, T: int = 100):
    if policy_type == "bc":
        policy = BCPolicy().to(device)
        policy.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        policy.eval()
        return policy
    elif policy_type == "diffusion":
        model = NoisePredictor().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        schedule = DDPMSchedule(T=T, device=device)
        return model, schedule
    else:
        raise ValueError(f"Unknown type: {policy_type}")


@torch.no_grad()
def get_action(policy, state_np: np.ndarray, policy_type: str, device: torch.device):
    state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
    if policy_type == "bc":
        action = policy(state).cpu().numpy()[0]
    else:
        model, schedule = policy
        action = schedule.sample(model, state).cpu().numpy()[0]
    return action


def evaluate(model_path: str, policy_type: str, difficulty: str,
             num_episodes: int = 50, render: bool = False, seed: int = 100,
             T: int = 100):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    policy = load_policy(model_path, policy_type, device, T=T)

    render_mode = "human" if render else None
    env = FlappyBirdEnv(difficulty=difficulty, render_mode=render_mode)

    rewards = []
    pipes_passed = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False

        while not done:
            action = get_action(policy, obs, policy_type, device)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)
        pipes_passed.append(env.score)

    env.close()

    rewards = np.array(rewards)
    pipes_passed = np.array(pipes_passed)
    print(f"\n{'='*50}")
    print(f"Policy: {policy_type.upper()} | Difficulty: {difficulty}")
    print(f"Episodes: {num_episodes}")
    print(f"Avg reward:       {rewards.mean():.2f} +/- {rewards.std():.2f}")
    print(f"Avg pipes passed: {pipes_passed.mean():.1f} +/- {pipes_passed.std():.1f}")
    print(f"Success rate (>=5 pipes): {(pipes_passed >= 5).mean()*100:.0f}%")
    print(f"{'='*50}\n")

    return rewards, pipes_passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--type", type=str, required=True, choices=["bc", "diffusion"])
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "hard"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--T", type=int, default=100)
    args = parser.parse_args()
    evaluate(args.model, args.type, args.difficulty, args.episodes, args.render, args.seed,
             args.T)


if __name__ == "__main__":
    main()
