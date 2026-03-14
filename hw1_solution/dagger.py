"""DAgger (Dataset Aggregation) for resolving bimodal expert ambiguity.

The key idea: roll out the current policy, but *relabel* each visited state
with a deterministic expert that always picks the upper gap (gap1).  Over
rounds the relabeled data resolves the bimodal ambiguity so that a simple
BC policy can succeed on hard mode.

Provided:
    - DeterministicExpert class skeleton (with reset).
    - run_dagger(): the full DAgger training loop.

TODO (students implement):
    - DeterministicExpert.act(): deterministic relabeling expert.
    - rollout_and_relabel(): roll out policy, relabel with expert, window
      into action chunks.
"""

import numpy as np
import torch

from expert import COMMIT_DIST
from flappy_bird_env import FlappyBirdEnv
from visualization import ChunkExecutor


class DeterministicExpert:
    """Expert that always picks gap1 (upper gap) in hard mode.

    Mimics the real Expert's behavior (hover at midpoint, commit when close,
    EMA smoothing) but replaces the random gap choice with a deterministic
    one.  This produces smooth, consistent relabeling.

    Behaviour:
        - While far from the pipe: target = midpoint of (gap1_y, gap2_y).
        - When dist < commit_dist: commit to gap1 (upper gap).
        - Apply EMA smoothing to the raw target.
        - Detect new pipes via a rounded (gap1, gap2) signature and reset
          commitment.

    Args:
        commit_dist: distance threshold to commit (default COMMIT_DIST).
        smoothing: EMA smoothing factor (default 0.15).
    """

    def __init__(self, commit_dist: float = COMMIT_DIST, smoothing: float = 0.15):
        self.commit_dist = commit_dist
        self.smoothing = smoothing
        self._last_gap_sig = None
        self._committed = False
        self._smooth_target = None

    def reset(self):
        self._last_gap_sig = None
        self._committed = False
        self._smooth_target = None

    def act(self, obs: np.ndarray) -> float:
        """Return a deterministic target y position in [0, 1].

        Args:
            obs: 4-D observation [dist_to_pipe, gap1_y, gap2_y, bird_y].

        Returns:
            Target y clipped to [0, 1].

        Implementation guide:
            1. Extract dist, gap1_y, gap2_y from obs.
            2. Detect new pipe: signature = (round(gap1_y, 3), round(gap2_y, 3)).
               If changed from self._last_gap_sig, reset self._committed and
               update self._last_gap_sig.
            3. Compute midpoint = (gap1_y + gap2_y) / 2.
            4. If not committed:
               - If dist < commit_dist: commit, raw_target = gap1_y (always
                 upper gap -- this is the key difference from the stochastic
                 Expert).
               - Else: raw_target = midpoint.
            5. If committed: raw_target = gap1_y.
            6. Apply EMA: if first call, smooth_target = raw_target; else
               smooth_target += smoothing * (raw_target - smooth_target).
            7. Return clipped to [0, 1].
        """
        dist = obs[0]
        gap1_y = obs[1]
        gap2_y = obs[2]

        gap_sig = (round(gap1_y, 3), round(gap2_y, 3))
        if self._last_gap_sig != gap_sig:
            self._committed = False
            self._last_gap_sig = gap_sig

        midpoint = (gap1_y + gap2_y) / 2.0

        if not self._committed:
            if dist < self.commit_dist:
                self._committed = True
                raw_target = float(gap1_y)
            else:
                raw_target = float(midpoint)
        else:
            raw_target = float(gap1_y)

        if self._smooth_target is None:
            self._smooth_target = raw_target
        else:
            self._smooth_target += self.smoothing * (
                raw_target - self._smooth_target)

        return float(np.clip(self._smooth_target, 0.0, 1.0))


@torch.no_grad()
def rollout_and_relabel(policy, difficulty, num_episodes, pipe_speed, seed,
                        action_chunk, device):
    """Roll out policy with chunk execution, relabel with deterministic expert.

    For each episode:
        1. Reset environment and helpers.
        2. Step through the episode using ChunkExecutor for the policy.
        3. At every step, query DeterministicExpert for the *expert* action
           (this is the relabeled action).
        4. Record (state, expert_action) pairs.
    After all episodes, window the per-step pairs into
    (state, action_chunk) training pairs.

    Args:
        policy: trained policy (callable, takes (1, state_dim) tensor).
        difficulty: "easy" or "hard".
        num_episodes: episodes to collect.
        pipe_speed: environment pipe speed.
        seed: base random seed.
        action_chunk: prediction horizon length K.
        device: torch device for policy inference.

    Returns:
        new_states: float32 array (N, 4).
        new_actions: float32 array (N, action_chunk).
    """
    policy.eval()
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed)
    if hasattr(policy, "set_env"):
        policy.set_env(env)
    executor = ChunkExecutor()
    det_expert = DeterministicExpert()
    new_states, new_actions = [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        if hasattr(policy, "reset"):
            policy.reset()
        executor.reset()
        det_expert.reset()
        done = False
        ep_states, ep_expert_actions = [], []

        while not done:
            if executor.needs_query():
                state_t = torch.tensor(obs, dtype=torch.float32,
                                       device=device).unsqueeze(0)
                pred = policy(state_t).cpu().numpy().flatten()
                executor.set_chunk(pred)

            optimal_action = det_expert.act(obs)

            ep_states.append(obs.copy())
            ep_expert_actions.append(optimal_action)

            action = executor.get_action()
            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            done = terminated or truncated

        for i in range(len(ep_states) - action_chunk + 1):
            new_states.append(ep_states[i])
            new_actions.append(ep_expert_actions[i:i + action_chunk])

    env.close()
    if not new_states:
        return np.zeros((0, 4), dtype=np.float32), \
               np.zeros((0, action_chunk), dtype=np.float32)
    return np.array(new_states, dtype=np.float32), \
           np.array(new_actions, dtype=np.float32)


def run_dagger(difficulty, initial_states, initial_actions, rounds,
               episodes_per_round, epochs, pipe_speed, seed,
               action_chunk, device, train_bc_fn,
               eval_episodes=100, lr=1e-3, batch_size=2048, verbose=False,
               initial_policy=None):
    """Run DAgger training loop with automatic relabeling.

    If *initial_policy* is provided it is used for evaluation and rollout
    in the first round (skipping a redundant BC training step).  In every
    subsequent round the policy is retrained on the aggregated data.

    Args:
        difficulty: "easy" or "hard".
        initial_states: initial expert states array.
        initial_actions: initial expert action-chunk array.
        rounds: number of DAgger rounds.
        episodes_per_round: rollout episodes per round.
        epochs: BC training epochs per round.
        pipe_speed: environment pipe speed.
        seed: base random seed.
        action_chunk: prediction horizon length K.
        device: torch device.
        train_bc_fn: callable(states, actions, epochs, batch_size, lr,
                     verbose, device) -> policy.
        eval_episodes: episodes for evaluation.
        lr: learning rate for BC.
        batch_size: batch size for BC.
        verbose: print training progress.
        initial_policy: optional pre-trained BC policy for round 1.

    Returns:
        policy: final trained policy.
        performance_means: list of mean episode lengths per round.
        performance_stds: list of std episode lengths per round.
    """
    from visualization import evaluate_policy

    all_states = initial_states.copy()
    all_actions = initial_actions.copy()
    performance_means = []
    performance_stds = []
    policy = initial_policy

    for rnd in range(1, rounds + 1):
        if policy is None:
            print(f"  Round {rnd}/{rounds}: Training on {len(all_states)} "
                  f"transitions...")
            policy = train_bc_fn(all_states, all_actions, epochs=epochs,
                                 verbose=verbose, batch_size=batch_size, lr=lr,
                                 device=device)
        else:
            print(f"  Round {rnd}/{rounds}: Using "
                  f"{'pretrained' if rnd == 1 else 'retrained'} "
                  f"policy ({len(all_states)} transitions in dataset)")

        avg_len, std_len = evaluate_policy(
            policy, difficulty, num_episodes=eval_episodes,
            pipe_speed=pipe_speed, seed=seed + rnd * 1000)
        performance_means.append(avg_len)
        performance_stds.append(std_len)
        print(f"    Evaluation: {avg_len:.1f} +/- {std_len:.1f} avg length "
              f"({eval_episodes} episodes)")

        print(f"    Collecting {episodes_per_round} episodes with "
              f"auto-relabeling...", end=" ", flush=True)
        new_states, new_actions = rollout_and_relabel(
            policy, difficulty, episodes_per_round,
            pipe_speed, seed + rnd * 10000,
            action_chunk=action_chunk, device=device)
        print(f"Got {len(new_states)} new transitions")

        if len(new_states) > 0:
            all_states = np.concatenate([new_states], axis=0)
            all_actions = np.concatenate([new_actions], axis=0)

        print(f"  Round {rnd}/{rounds}: Retraining on {len(all_states)} "
              f"transitions...")
        policy = train_bc_fn(all_states, all_actions, epochs=epochs,
                             verbose=verbose, batch_size=batch_size, lr=lr,
                             device=device)

    return policy, performance_means, performance_stds
