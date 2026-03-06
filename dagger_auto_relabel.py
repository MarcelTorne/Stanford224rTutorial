"""DAgger with automatic relabeling based on trajectory outcomes.

Instead of querying an expert, we automatically compute the "perfect action"
by analyzing which actions would have led to better outcomes.
"""

import numpy as np
import torch
from collect_data import compute_action
from flappy_bird_env import FlappyBirdEnv, SCREEN_H


@torch.no_grad()
def collect_and_relabel_trajectories(policy, difficulty, num_episodes, pipe_speed, seed, device):
    """
    Collect trajectories and automatically relabel with corrected actions.

    Strategy:
    1. Run policy to collect trajectories
    2. For failed episodes, compute what action would have been better
    3. Relabel states with corrected actions based on what went wrong
    """
    policy.eval()
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed)

    all_states = []
    all_actions = []
    all_corrected_actions = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)

        trajectory = []
        done = False

        while not done:
            state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = policy(state_t).detach().cpu().numpy()[0]

            trajectory.append({
                'state': obs.copy(),
                'action': action.copy(),
            })

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if terminated:
            corrected_traj = compute_corrections(trajectory, env, difficulty)
        else:
            corrected_traj = trajectory

        for step in corrected_traj:
            all_states.append(step['state'])
            all_corrected_actions.append([step.get('corrected_action', step['action'])])

    env.close()
    return np.array(all_states, dtype=np.float32), np.array(all_corrected_actions, dtype=np.float32)


def compute_corrections(trajectory, env, difficulty):
    """
    Compute corrected actions (thrust) for a failed trajectory.

    Strategy:
    - Look at where the bird crashed (too high/low)
    - Backtrack and compute thrust toward the safer gap
    """
    if len(trajectory) == 0:
        return trajectory

    final_step = trajectory[-1]
    final_obs = final_step['state']
    bird_y_final = final_obs[3]

    crashed_high = bird_y_final < 0.3
    crashed_low = bird_y_final > 0.7

    correction_window = max(1, len(trajectory) // 5)

    for i in range(len(trajectory) - correction_window, len(trajectory)):
        step = trajectory[i]
        obs = step['state']

        bird_y = obs[3]

        if difficulty == "easy":
            target_gap = obs[1]
        else:
            gap1_y = obs[1]
            gap2_y = obs[2]

            if crashed_high:
                target_gap = max(gap1_y, gap2_y)
            elif crashed_low:
                target_gap = min(gap1_y, gap2_y)
            else:
                target_gap = gap1_y if abs(gap1_y - 0.5) < abs(gap2_y - 0.5) else gap2_y

        target_gap = target_gap * 0.7 + 0.5 * 0.3
        step['corrected_action'] = compute_action(bird_y, target_gap)

    return trajectory


@torch.no_grad()
def evaluate_with_details(policy, difficulty, num_episodes, pipe_speed, seed, device):
    """Evaluate and return detailed statistics."""
    policy.eval()
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed)

    episode_lengths = []
    crash_reasons = {'high': 0, 'low': 0, 'timeout': 0}

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False

        while not done:
            state_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = policy(state_t).detach().cpu().numpy()[0]
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        episode_lengths.append(env.step_count)

        if truncated:
            crash_reasons['timeout'] += 1
        elif env.bird_y < SCREEN_H * 0.3:
            crash_reasons['high'] += 1
        else:
            crash_reasons['low'] += 1

    env.close()

    return {
        'mean': np.mean(episode_lengths),
        'std': np.std(episode_lengths),
        'crash_reasons': crash_reasons
    }
