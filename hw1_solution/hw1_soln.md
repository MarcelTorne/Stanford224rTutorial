# HW1 Solutions

## `networks.py`
### BCPolicy
```python
def __init__(self, state_dim: int = 4, action_dim: int = 1, hidden: int = 256):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(state_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, action_dim),
        nn.Sigmoid(),
    )

def forward(self, state):
    return self.net(state)
```

## `losses.py`
### bc_loss
```python
def bc_loss(policy, s_batch, a_batch):
    pred = policy(s_batch)
    return nn.MSELoss()(pred, a_batch)
```

### flow_matching_loss
```python
def flow_matching_loss(policy, s_batch, a_batch):
    bsz = s_batch.size(0)
    t = torch.rand(bsz, device=s_batch.device)
    x_t, velocity = policy.schedule.interpolate(a_batch, t)
    pred_v = policy(x_t, s_batch, t)
    return nn.MSELoss()(pred_v, velocity)
```

## `expert.py`
### Expert.act
```python
def act(self, obs, difficulty):
    dist = obs[0]
    gap1_y = obs[1]
    gap2_y = obs[2]

    if difficulty == "easy":
        raw_target = float(gap1_y)
    else:
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
                raw_target = float(midpoint)

        if self._committed:
            raw_target = float(gap1_y if self.target_gap_idx == 0 else gap2_y)

    if self._smooth_target is None:
        self._smooth_target = raw_target
    else:
        self._smooth_target += self.smoothing * (raw_target - self._smooth_target)

    return float(np.clip(self._smooth_target, 0.0, 1.0))
```

## `dagger.py`
### DeterministicExpert.act
```python
def act(self, obs):
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
```

### rollout_and_relabel
```python
def rollout_and_relabel(policy, difficulty, num_episodes, pipe_speed, seed,
                        action_chunk, device):
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
```
