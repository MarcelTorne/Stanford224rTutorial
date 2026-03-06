"""Check if diffusion/flow actions oscillate between modes during a trajectory."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from collect_data import Expert
from flappy_bird_env import FlappyBirdEnv
from train_bc import BCPolicy
from train_flow_matching import VelocityNet, sample as flow_sample

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")


def collect_expert_data(difficulty, num_episodes, seed=0):
    env = FlappyBirdEnv(difficulty=difficulty)
    expert = Expert()
    all_states, all_actions = [], []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        expert.reset()
        done = False
        while not done:
            a = expert.act(obs, difficulty)
            all_states.append(obs.copy())
            all_actions.append([a])
            obs, _, terminated, truncated, _ = env.step(np.array([a]))
            done = terminated or truncated
    env.close()
    return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.float32)


def train_flow(states, actions, epochs=200):
    loader = DataLoader(
        TensorDataset(torch.tensor(states), torch.tensor(actions)),
        batch_size=256, shuffle=True)
    model = VelocityNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    for epoch in range(epochs):
        for s, a in loader:
            s, a = s.to(DEVICE), a.to(DEVICE)
            x_0 = torch.randn_like(a)
            t = torch.rand(s.size(0), device=DEVICE)
            t_exp = t.unsqueeze(-1)
            x_t = (1 - t_exp) * x_0 + t_exp * a
            target_v = a - x_0
            pred_v = model(x_t, s, t)
            loss = nn.MSELoss()(pred_v, target_v)
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{epochs}")
    return model


@torch.no_grad()
def run_episode_and_log(policy_fn, difficulty, seed, label):
    """Run one episode and log all actions."""
    env = FlappyBirdEnv(difficulty=difficulty)
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0
    actions_log = []
    states_log = []
    while not done:
        s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        action = policy_fn(s).cpu().numpy()[0]
        actions_log.append(action[0])
        states_log.append(obs.copy())
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1

    actions_log = np.array(actions_log)
    states_log = np.array(states_log)

    print(f"\n--- {label} (seed={seed}) ---")
    print(f"  Survived {env.step_count} steps ({'crashed' if terminated else 'timeout'})")

    # Find moments near pipes and show consecutive actions
    dists = states_log[:, 0]
    near_pipe_starts = []
    in_approach = False
    for i in range(len(dists)):
        if dists[i] < 0.2 and not in_approach:
            in_approach = True
            near_pipe_starts.append(i)
        elif dists[i] >= 0.3:
            in_approach = False

    # Show first 3 pipe approaches
    for pipe_idx, start in enumerate(near_pipe_starts[:3]):
        end = min(start + 30, len(actions_log))
        print(f"  Pipe approach {pipe_idx+1} (steps {start}-{end}):")
        for i in range(start, end):
            dist = states_log[i, 0]
            gap1 = states_log[i, 1]
            gap2 = states_log[i, 2]
            bird = states_log[i, 3]
            a = actions_log[i]
            marker = ""
            if abs(a - gap1) < 0.03:
                marker = " <- gap1"
            elif abs(a - gap2) < 0.03:
                marker = " <- gap2"
            elif abs(a - (gap1+gap2)/2) < 0.03:
                marker = " <- midpoint"
            print(f"    step={i:3d} dist={dist:+.3f} gap1={gap1:.3f} gap2={gap2:.3f} "
                  f"bird={bird:.3f} action={a:.3f}{marker}")

    # Action jitter stats
    action_diffs = np.abs(np.diff(actions_log))
    print(f"  Action jitter: mean={action_diffs.mean():.4f}, max={action_diffs.max():.4f}")

    env.close()
    return env.step_count


def main():
    states, actions = collect_expert_data("hard", 100, seed=0)
    print(f"Collected {len(states)} transitions")

    print("\nTraining flow matching...")
    flow_model = train_flow(states, actions, epochs=200)
    flow_model.eval()

    print("Training BC...")
    loader = DataLoader(
        TensorDataset(torch.tensor(states), torch.tensor(actions)),
        batch_size=256, shuffle=True)
    bc = BCPolicy().to(DEVICE)
    opt = torch.optim.Adam(bc.parameters(), lr=1e-3)
    for _ in range(50):
        for s, a in loader:
            s, a = s.to(DEVICE), a.to(DEVICE)
            loss = nn.MSELoss()(bc(s), a)
            opt.zero_grad(); loss.backward(); opt.step()
    bc.eval()

    # Run episodes and compare
    for seed in [42, 100, 200]:
        run_episode_and_log(lambda s: bc(s), "hard", seed, "BC")
        run_episode_and_log(
            lambda s: flow_sample(flow_model, s, 20), "hard", seed, "Flow Matching")


if __name__ == "__main__":
    main()
