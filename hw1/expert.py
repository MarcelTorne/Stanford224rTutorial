"""Expert policy and data collection for Flappy Bird.

The expert outputs a target y position (normalised 0-1) for each timestep.
The environment's internal PD controller converts the target into thrust.

Provided:
    - collect_expert_data(): collects expert demonstrations and windows them
      into (state, action_chunk) training pairs.

TODO (students implement):
    - Expert.act(): expert targeting logic for easy and hard modes.
"""

import numpy as np
from flappy_bird_env import FlappyBirdEnv

COMMIT_DIST = 0.18  # normalised pipe distance at which the expert picks a gap


class Expert:
    """Expert that outputs target y positions (normalised 0-1).

    Easy mode: target = gap centre (gap1_y from observation).

    Hard mode: target = midpoint between the two gaps while far away.
    When the bird gets within ``commit_dist`` of the pipe, randomly pick
    one of the two gaps and target there for the remainder of that pipe.
    Reset the commitment when a new pipe appears (detected by a change in
    gap positions).

    Actions are temporally smoothed with an EMA to avoid discontinuous jumps:
        smooth_target += smoothing * (raw_target - smooth_target)

    Relevant observation indices:
        obs[0] = dist_to_pipe  (normalised)
        obs[1] = gap1_y        (normalised)
        obs[2] = gap2_y        (normalised)
        obs[3] = bird_y        (normalised)

    Args:
        commit_dist: distance threshold to commit to a gap (default 0.18).
        smoothing: EMA smoothing factor (default 0.15).
    """

    def __init__(self, commit_dist: float = COMMIT_DIST, smoothing: float = 0.15):
        self.commit_dist = commit_dist
        self.smoothing = smoothing
        self.target_gap_idx = None
        self._last_gap_sig = None
        self._committed = False
        self._smooth_target = None

    def reset(self):
        self.target_gap_idx = None
        self._last_gap_sig = None
        self._committed = False
        self._smooth_target = None

    def act(self, obs: np.ndarray, difficulty: str) -> float:
        """Return target y position in [0, 1].

        Args:
            obs: 4-D observation [dist_to_pipe, gap1_y, gap2_y, bird_y].
            difficulty: "easy" or "hard".

        Returns:
            Target y position clipped to [0, 1].

        Implementation guide:
            1. Extract dist, gap1_y, gap2_y from obs.
            2. Easy mode: raw_target = gap1_y.
            3. Hard mode:
               a. Detect new pipe: compute a signature from (gap1_y, gap2_y)
                  rounded to 3 decimals. If it changed, reset commitment.
               b. Compute midpoint = (gap1_y + gap2_y) / 2.
               c. If not committed and dist < commit_dist:
                  randomly pick target_gap_idx in {0, 1}, mark committed.
               d. If not committed: raw_target = midpoint.
               e. If committed: raw_target = gap1_y if idx==0 else gap2_y.
            4. Apply EMA smoothing to raw_target.
            5. Return clipped to [0, 1].
        """
        # ============================================================
        # TODO: Implement expert targeting logic.
        # Extract obs[0], obs[1], obs[2]. Handle easy vs hard mode.
        # Apply EMA smoothing. Return clipped result.
        # ============================================================
        raise NotImplementedError("TODO: Implement Expert.act")


def collect_expert_data(difficulty, num_episodes, action_chunk,
                        pipe_speed=3.0, seed=0):
    """Collect expert demos step-by-step, then window into action chunks.

    Training pairs are (s_t, [a_t, a_{t+1}, ..., a_{t+K-1}]) where
    K = action_chunk.

    Args:
        difficulty: "easy" or "hard".
        num_episodes: number of episodes to collect.
        action_chunk: prediction horizon length K.
        pipe_speed: environment pipe speed.
        seed: base random seed.

    Returns:
        states: float32 array of shape (N, 4).
        actions: float32 array of shape (N, action_chunk).
    """
    env = FlappyBirdEnv(difficulty=difficulty, pipe_speed=pipe_speed)
    expert = Expert()
    all_states, all_actions = [], []
    all_steps = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        expert.reset()
        done = False
        ep_states, ep_actions = [], []
        while not done:
            action = expert.act(obs, difficulty)
            ep_states.append(obs.copy())
            ep_actions.append(action)
            obs, _, terminated, truncated, _ = env.step(np.array([action]))
            done = terminated or truncated
        all_steps.append(len(ep_states))
        for i in range(len(ep_states) - action_chunk + 1):
            all_states.append(ep_states[i])
            all_actions.append(ep_actions[i:i + action_chunk])
    print(f"Average steps: {np.mean(all_steps):.1f}")
    print(f"Chunk={action_chunk}: {len(all_states)} chunk pairs "
          f"from {num_episodes} episodes")
    env.close()
    return (np.array(all_states, dtype=np.float32),
            np.array(all_actions, dtype=np.float32))
