"""Collect expert demonstrations for the Flappy Bird environment.

The expert outputs a target y position (normalised 0-1) for each timestep.
In easy mode, it targets the single gap centre.
In hard mode, it targets the midpoint between gaps, then at a commit distance
randomly picks one gap and targets there.

The environment's internal PD controller converts the target position into
thrust.  The expert is simple: it just says WHERE it wants the bird to be.

Usage:
    python collect_data.py --difficulty easy --num_episodes 500 --output data/easy_expert.npz
    python collect_data.py --difficulty hard --num_episodes 500 --output data/hard_expert.npz
"""

import argparse
import os

import numpy as np
from flappy_bird_env import FlappyBirdEnv, SCREEN_H

COMMIT_DIST = 0.30  # normalised pipe distance at which the expert picks a gap


class Expert:
    """Expert that outputs target y positions (normalised 0-1).

    Easy mode: target = gap centre.
    Hard mode: target = midpoint between gaps, then at commit_dist randomly
    pick one gap and target there.
    """

    def __init__(self, commit_dist: float = COMMIT_DIST):
        self.commit_dist = commit_dist
        self.target_gap_idx = None
        self._last_gap_sig = None
        self._committed = False

    def reset(self):
        self.target_gap_idx = None
        self._last_gap_sig = None
        self._committed = False

    def act(self, obs: np.ndarray, difficulty: str) -> float:
        """Return target y position in [0, 1]."""
        dist = obs[0]
        gap1_y = obs[1]
        gap2_y = obs[2]

        if difficulty == "easy":
            return float(gap1_y)

        # --- Hard mode ---
        gap_sig = (round(gap1_y, 3), round(gap2_y, 3))
        if self._last_gap_sig != gap_sig:
            self._committed = False
            self.target_gap_idx = None
            self._last_gap_sig = gap_sig

        midpoint = (gap1_y + gap2_y) / 2.0

        if not self._committed:
            if dist < self.commit_dist:
                self.target_gap_idx = np.random.choice([0, 1])
                self._committed = True
            else:
                return float(midpoint)

        target_y = gap1_y if self.target_gap_idx == 0 else gap2_y
        return float(target_y)


def compute_action(bird_y: float, target_y: float) -> float:
    """Compute action (target y position) to move bird toward target_y."""
    return float(target_y)


def compute_action_gravity(bird_y: float, target_y: float, bird_vel: float = 0.0) -> float:
    """Compute action accounting for gravity (same as compute_action for PD-controlled env)."""
    return float(target_y)


def collect(difficulty: str, num_episodes: int, seed: int = 0,
            commit_dist: float = COMMIT_DIST, pipe_speed: float = 3.0):
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed)
    expert = Expert(commit_dist=commit_dist)
    all_states, all_actions = [], []
    total_reward = 0.0
    total_pipes = 0

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        expert.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = expert.act(obs, difficulty)
            all_states.append(obs.copy())
            all_actions.append([action])
            obs, reward, terminated, truncated, _ = env.step(np.array([action]))
            ep_reward += reward
            done = terminated or truncated
        total_reward += ep_reward
        total_pipes += env.score

    env.close()
    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    return states, actions, total_reward / num_episodes, total_pipes / num_episodes


def main():
    parser = argparse.ArgumentParser(description="Collect expert demonstrations")
    parser.add_argument("--difficulty", type=str, default="easy", choices=["easy", "hard"])
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--commit_dist", type=float, default=COMMIT_DIST)
    parser.add_argument("--pipe_speed", type=float, default=3.0)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"data/{args.difficulty}_expert.npz"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Collecting {args.num_episodes} episodes on {args.difficulty} mode "
          f"(pipe_speed={args.pipe_speed})...")
    states, actions, avg_reward, avg_pipes = collect(
        args.difficulty, args.num_episodes, args.seed, args.commit_dist,
        pipe_speed=args.pipe_speed,
    )
    np.savez(args.output, states=states, actions=actions)
    print(f"Saved {len(states)} transitions to {args.output}")
    print(f"Avg reward: {avg_reward:.2f}, Avg pipes passed: {avg_pipes:.1f}")


if __name__ == "__main__":
    main()
