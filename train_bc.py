"""Behavior Cloning with MSE regression.

Usage:
    python train_bc.py --data data/easy_expert.npz --output models/bc_easy.pt
    python train_bc.py --data data/hard_expert.npz --output models/bc_hard.pt
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BCPolicy(nn.Module):
    """Simple MLP: state (4) -> target y position (1)."""

    def __init__(self, state_dim: int = 4, action_dim: int = 1, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Sigmoid(),  # output in [0, 1] (target y position)
        )

    def forward(self, state):
        return self.net(state)


def train(data_path: str, output_path: str, epochs: int = 200, batch_size: int = 256,
          lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    data = np.load(data_path)
    states = torch.tensor(data["states"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.float32)

    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/n:.6f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(policy.state_dict(), output_path)
    print(f"Saved BC policy to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/bc.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
