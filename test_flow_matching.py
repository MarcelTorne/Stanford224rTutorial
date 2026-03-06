"""Quick test: BC vs Flow Matching on hard mode."""

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
print(f"Device: {DEVICE}")


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


def train_flow(states, actions, epochs=200, num_inference_steps=20):
    loader = DataLoader(
        TensorDataset(torch.tensor(states), torch.tensor(actions)),
        batch_size=256, shuffle=True)
    model = VelocityNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    for epoch in range(epochs):
        total_loss = 0.0; n = 0
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
            total_loss += loss.item() * s.size(0); n += s.size(0)
        if (epoch + 1) % 50 == 0:
            print(f"    Flow epoch {epoch+1}/{epochs} loss={total_loss/n:.6f}")
    return model, num_inference_steps


@torch.no_grad()
def evaluate(policy_fn, difficulty, num_episodes=30, seed=9000):
    env = FlappyBirdEnv(difficulty=difficulty)
    lengths = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action = policy_fn(s).cpu().numpy()[0]
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        lengths.append(env.step_count)
    env.close()
    return np.mean(lengths), np.std(lengths)


def main():
    n_demos = 100
    print(f"Collecting {n_demos} expert demos on hard mode...")
    states, actions = collect_expert_data("hard", n_demos, seed=0)
    print(f"  {len(states)} transitions\n")

    # BC baseline
    print("Training BC (50 epochs)...")
    loader = DataLoader(
        TensorDataset(torch.tensor(states), torch.tensor(actions)),
        batch_size=256, shuffle=True)
    bc = BCPolicy().to(DEVICE)
    opt = torch.optim.Adam(bc.parameters(), lr=1e-3)
    for ep in range(50):
        for s, a in loader:
            s, a = s.to(DEVICE), a.to(DEVICE)
            loss = nn.MSELoss()(bc(s), a)
            opt.zero_grad(); loss.backward(); opt.step()
    bc.eval()
    bc_mean, bc_std = evaluate(lambda s: bc(s), "hard")
    print(f"  BC: {bc_mean:.1f} ± {bc_std:.1f}\n")

    # Flow matching with different inference steps
    for n_steps in [10, 20, 50]:
        print(f"Training Flow Matching (200 epochs, {n_steps} inference steps)...")
        model, _ = train_flow(states, actions, epochs=200, num_inference_steps=n_steps)
        model.eval()
        fm_mean, fm_std = evaluate(
            lambda s, m=model, ns=n_steps: flow_sample(m, s, ns), "hard")
        print(f"  Flow ({n_steps} steps): {fm_mean:.1f} ± {fm_std:.1f}\n")

    print("Done!")


if __name__ == "__main__":
    main()
