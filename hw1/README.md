# CS224R Homework 1: Imitation Learning on Flappy Bird

This folder contains the starter code for Homework 1 of CS224R (Deep Reinforcement Learning), focused on imitation learning with action chunking in a custom Flappy Bird environment.

The goal of this README is to be readable even if you are new to imitation learning.

## 1. What You Are Building

You are implementing a standard imitation learning pipeline used in modern robot learning research:

1. `BC` (Behavior Cloning): learn a direct state-to-action mapping from expert demonstrations via MSE regression.
2. `Flow Matching` (Conditional OT-CFM): learn a generative model of the action distribution that can capture multimodal expert behavior.
3. `DAgger` (Dataset Aggregation): iteratively relabel on-policy rollouts with a deterministic expert to resolve distribution shift and multimodal ambiguity.
4. `Evaluation`: measure how long the learned policy survives (average episode length out of 1000 max steps).

Important: this is starter code for a class assignment. Core logic is intentionally left as TODOs for you to implement.

## 2. Current Project Status (What Is and Is Not Implemented)

Implemented (provided):
- Flappy Bird Gymnasium environment with easy/hard modes and PD-controlled physics.
- Expert data collection with action-chunk windowing.
- Full training pipeline orchestration, evaluation, and video recording.
- Diffusion Policy, DDPM schedule, and U-Net architecture.
- Gaussian BC policy architecture (bonus, not required).
- DAgger outer loop (`run_dagger`).
- All visualization and policy wrapper utilities.

Not implemented (you need to write these):
- `networks.py`: `BCPolicy.__init__`, `BCPolicy.forward`, `FlowMatchingSchedule.interpolate`, and `FlowMatchingSchedule.sample` raise `NotImplementedError`.
- `losses.py`: `bc_loss` and `flow_matching_loss` raise `NotImplementedError`.
- `expert.py`: `Expert.act` raises `NotImplementedError`.
- `dagger.py`: `DeterministicExpert.act` and `rollout_and_relabel` raise `NotImplementedError`.

## 3. Flappy Bird Task Overview

The environment is a physics-based Flappy Bird where the agent controls a target y-position (normalised to [0, 1]) rather than discrete flap/no-flap. A PD controller converts the target into thrust internally, creating momentum-based dynamics that require anticipation.

### Observation (4-D, all normalised)

| Index | Name | Description |
|-------|------|-------------|
| 0 | `dist_to_pipe` | Horizontal distance to next pipe |
| 1 | `gap1_y` | Vertical position of gap 1 (upper gap in hard mode) |
| 2 | `gap2_y` | Vertical position of gap 2 (same as gap1 in easy mode) |
| 3 | `bird_y` | Current bird vertical position |

### Action

A single float in [0, 1] representing the target y-position. The environment's PD controller (with gains Kp=3.5, Kd=1.2) converts this to thrust.

### Difficulty Modes

- **Easy mode**: each pipe has one gap opening. The expert demonstration distribution is unimodal, so simple BC regression works.
- **Hard mode**: each pipe has two gap openings separated by 150px. The expert randomly commits to one gap when close, creating bimodal demonstrations. BC regression averages the two modes and crashes into the wall between them.

### Action Chunking

Policies predict `ACTION_CHUNK=20` future target positions at once. During rollout, only the first `EXECUTE_STEPS=10` are executed before re-querying the policy (receding horizon). This temporal structure lets generative models (flow matching) capture correlated multi-step plans.

### Episode Termination

- **Collision**: bird hits a pipe wall or the screen boundary (reward = -1).
- **Timeout**: episode reaches 1000 steps (the goal).

## 4. Folder Map

```text
├── README.md                    # This file
├── installation.md              # Environment setup instructions
├── requirements.txt             # Python dependencies
├── assets/                      # Sprites for Flappy Bird rendering
│   ├── background.png
│   ├── background2.png
│   ├── pipe.png
│   ├── robobird_up.png
│   └── robobird_down.png
│
├── main.py                      # Training pipeline & CLI entrypoint (read-only)
├── networks.py                  # Network architectures               [TODO: BCPolicy, FlowMatchingSchedule]
├── losses.py                    # Loss functions                      [TODO: bc_loss, flow_matching_loss]
├── expert.py                    # Expert policy + data collection     [TODO: Expert.act]
├── dagger.py                    # DAgger relabeling + training loop   [TODO: DeterministicExpert.act, rollout_and_relabel]
├── flappy_bird_env.py           # Gymnasium environment (read-only)
├── visualization.py             # Evaluation, video, policy wrappers (read-only)
│
├── models/                      # Saved model checkpoints (generated at runtime)
└── plots/                       # Generated plots and videos (generated at runtime)
```

## 5. Setup

See [installation.md](installation.md) for detailed instructions.

### 5.1 Python Environment (recommended)

```bash
conda create -n cs224r python=3.11
conda activate cs224r
pip install -r requirements.txt
```

### 5.2 Dependencies

```text
torch
numpy
gymnasium
pygame
matplotlib
imageio[ffmpeg]
```

Notes:
- A CUDA-capable GPU is not strictly required but speeds up training significantly.
- The code supports automatic device selection: CUDA > MPS (Apple Silicon) > CPU.
- If you encounter display-related errors when rendering videos, set `export SDL_VIDEODRIVER=dummy`.

## 6. Key Concepts

This section briefly explains the algorithms so you know what you are implementing.

### 6.1 Behavior Cloning (BC)

The simplest approach: train a neural network to predict the expert's action given the current state, using mean squared error. Works well when the expert distribution is unimodal (easy mode) but fails when demonstrations are multimodal (hard mode) because MSE regression averages the modes.

### 6.2 Flow Matching (Conditional OT-CFM)

A generative model that learns a velocity field transporting noise to the data distribution. At training time, the network predicts the velocity `v = x_1 - noise` at interpolated points `x_t = (1-t)*noise + t*x_1`. At inference, an Euler ODE integrator follows the learned velocity from pure noise (t=0) to data (t=1). Unlike BC, this can represent multimodal distributions.

Compare with the provided DDPM schedule: DDPM adds scaled Gaussian noise and predicts that noise; flow matching linearly interpolates between noise and data and predicts the velocity. Flow matching sampling is simpler -- just Euler integration, no alpha/beta schedule.

### 6.3 DAgger (Dataset Aggregation)

An interactive imitation learning algorithm that addresses distribution shift. Each round: (1) roll out the current policy, (2) relabel visited states with a deterministic expert (always picks the upper gap), (3) aggregate the relabeled data with existing data, (4) retrain BC. Over rounds, the data distribution shifts from bimodal to unimodal, allowing BC to succeed.

## 7. Your Tasks

Every function you need to implement is marked with `raise NotImplementedError("TODO: ...")`. Search for `TODO` across the codebase to find them all.

Look for sections marked with `HW1` in other files to see how your edits will be used. Some files you may find relevant for reference:
- [main.py](main.py) -- training pipeline and evaluation (read-only)
- [visualization.py](visualization.py) -- evaluation and video utilities (read-only)

See the homework PDF for more details.

### Part 1: Behavior Cloning on Easy Mode

Files: `networks.py`, `losses.py`, `expert.py`

1. **`BCPolicy.__init__`** and **`BCPolicy.forward`** in `networks.py`: build a 3-layer MLP with the architecture described in the docstring (Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid).
2. **`bc_loss`** in `losses.py`: MSE between the policy's predicted actions and expert actions.
3. **`Expert.act`** in `expert.py`: easy mode targets `gap1_y`; hard mode hovers at the midpoint of both gaps and randomly commits to one when within `commit_dist`; applies EMA smoothing.

Verify:

```bash
python main.py --method bc --env easy
```

Expected: ~900-1000 average episode length.

### Part 2: BC Fails on Hard Mode

No new code needed. Run BC on hard mode and observe the failure:

```bash
python main.py --method bc --env hard
```

Expected: ~100-200 average steps (crashes because MSE averages the two gaps).

### Part 3: Flow Matching on Hard Mode

Files: `networks.py`, `losses.py`

1. **`FlowMatchingSchedule.interpolate`** in `networks.py`: given clean data `x_1` and timesteps `t in [0,1]`, sample noise, compute the linear interpolation `x_t = (1-t)*noise + t*x_1`, and return `(x_t, velocity)` where `velocity = x_1 - noise`. Compare with the provided `DDPMSchedule.q_sample` to see how the two approaches differ.
2. **`FlowMatchingSchedule.sample`** in `networks.py`: starting from pure noise, run Euler ODE integration from `t=0` to `t=1` using the learned velocity model. At each step: `x = x + model(x, state, t) * dt`. Clamp the final output to [0, 1]. Compare with the provided `DDPMSchedule.sample` -- flow matching sampling is simpler (no alpha/beta schedule).
3. **`flow_matching_loss`** in `losses.py`: sample random `t ~ Uniform(0,1)`, interpolate with `policy.schedule.interpolate(a_batch, t)`, predict velocity with `policy(x_t, s_batch, t)`, return MSE between predicted and target velocity.

Verify:

```bash
python main.py --method flow_matching --env hard
```

Expected: ~800-1000 average episode length.

### Part 4: DAgger on Hard Mode

File: `dagger.py`

1. **`DeterministicExpert.act`**: same logic as `Expert.act` in hard mode but always picks gap1 (upper gap) instead of random selection. This is the key difference that makes DAgger resolve the bimodal ambiguity.
2. **`rollout_and_relabel`**: roll out the current policy using `ChunkExecutor`, query `DeterministicExpert` at every step for the relabeled action, window results into action-chunk training pairs (follow the same windowing pattern as `collect_expert_data` in `expert.py`).

Verify:

```bash
python main.py --method dagger --env hard
```

Expected: progressive improvement from ~100-200 to ~600-1000 average steps over 5 rounds.

## 8. Running the Pipeline

Tip: use specific `--method` and `--env` flags to run one part at a time instead of the full pipeline.

### Individual Methods

```bash
python main.py --method bc --env easy
python main.py --method bc --env hard
python main.py --method flow_matching --env hard
python main.py --method dagger --env hard
```

### Full Pipeline (all methods, all environments)

```bash
python main.py
```

### Output

Each run creates timestamped directories:
- `plots/<timestamp>/` -- comparison charts, DAgger learning curves, and `.mp4` episode videos.
- `models/<timestamp>/` -- saved PyTorch model checkpoints (`.pt` files).

## 9. Environment Details

### Physics

| Parameter | Value | Description |
|-----------|-------|-------------|
| Gravity | 0.5 px/step^2 | Constant downward acceleration |
| Thrust scale | 2.5 px/step^2 | Per-unit upward acceleration |
| Max velocity | 10.0 px/step | Velocity clamp |
| PD Kp | 3.5 | Proportional gain (position error) |
| PD Kd | 1.2 | Derivative gain (velocity damping) |
| Pipe speed | 3.0 px/step | Horizontal pipe scroll rate |
| Pipe gap size | 75 px | Vertical size of each opening |
| Hard gap separation | 150 px | Distance between gap centers in hard mode |
| Screen size | 800 x 448 px | Width x Height |
| Max steps | 1000 | Episode timeout |

### Expert Behavior

The expert uses a commitment-based strategy with EMA smoothing (factor 0.15):
- **Easy mode**: always targets `gap1_y`.
- **Hard mode**: hovers at the midpoint of both gaps while far away. When within `COMMIT_DIST=0.18` normalised distance, randomly commits to gap1 or gap2 for the remainder of that pipe. Resets commitment when a new pipe is detected (via a rounded gap-position signature).

## 10. Expected Results

| Method | Easy Mode (avg steps) | Hard Mode (avg steps) |
|--------|----------------------|----------------------|
| Expert | ~1000 | ~1000 |
| BC (MSE) | ~900-1000 | ~100-200 (crashes) |
| Flow Matching (OT-CFM) | ~900-1000 | ~800-1000 |
| DAgger (5 rounds) | -- | ~600-1000 |

The critical insight: BC regression fails on hard mode because it averages two modes, targeting the wall between the gaps. Flow matching succeeds because it can represent the full bimodal distribution. DAgger succeeds by gradually replacing bimodal data with consistent unimodal relabeling.

## 11. Milestones

1. Implement `Expert.act`, `BCPolicy`, and `bc_loss`. Verify BC works on easy mode (~900-1000 steps).
2. Confirm BC fails on hard mode (~100-200 steps). Understand why MSE regression cannot handle bimodal data.
3. Implement `FlowMatchingSchedule.interpolate`, `FlowMatchingSchedule.sample`, and `flow_matching_loss`. Verify flow matching succeeds on hard mode (~800-1000 steps).
4. Implement `DeterministicExpert.act` and `rollout_and_relabel`. Verify DAgger progressively improves on hard mode.
5. Compare all methods. Examine generated plots and videos.

## 12. Troubleshooting

**`NotImplementedError` during training:**
- Expected until you complete the TODO functions listed in Section 2.

**Out-of-memory (OOM):**
- Reduce `BC_BATCH_SIZE` in `main.py` (default 2048).
- The U-Net models are lightweight; OOM is unlikely unless running on very constrained hardware.

**Pygame display errors / `SDL` issues:**
- Set `export SDL_VIDEODRIVER=dummy` before running.
- Ensure `pygame` is installed: `pip install pygame`.

**Video encoding errors:**
- Ensure `imageio[ffmpeg]` is installed. If ffmpeg is missing: `pip install imageio[ffmpeg]`.

**BC works on easy but scores ~100-200 on hard:**
- This is expected behavior, not a bug. See Section 10 for why.

**DAgger not improving:**
- Verify `rollout_and_relabel` correctly uses `ChunkExecutor` for the policy but `DeterministicExpert` for the relabeled actions.
- Check that you are aggregating (concatenating) new data with existing data each round, not replacing it.

**Slow training:**
- Training runs on CPU by default if no GPU is detected. Each method takes a few minutes on CPU.

## 13. Quick Command Cheat Sheet

```bash
# Setup
conda create -n cs224r python=3.11 && conda activate cs224r
pip install -r requirements.txt

# Part 1: BC on easy (after implementing BCPolicy, bc_loss, Expert.act)
python main.py --method bc --env easy

# Part 2: BC on hard (observe failure)
python main.py --method bc --env hard

# Part 3: Flow matching on hard (after implementing FlowMatchingSchedule + flow_matching_loss)
python main.py --method flow_matching --env hard

# Part 4: DAgger on hard (after implementing DeterministicExpert.act, rollout_and_relabel)
python main.py --method dagger --env hard

# Full pipeline (all methods, all envs)
python main.py
```

## 14. Final Notes

This starter code is intentionally structured so you can focus on algorithm implementation, not project plumbing.

If you are unsure where to begin, implement in this order:
1. `expert.py::Expert.act` -- understand the environment and expert behavior first.
2. `networks.py::BCPolicy` -- a simple 3-layer MLP.
3. `losses.py::bc_loss` -- one-line MSE loss.
4. Verify Parts 1 and 2 work before moving on.
5. `networks.py::FlowMatchingSchedule.interpolate` and `FlowMatchingSchedule.sample` -- compare with the provided `DDPMSchedule` to see the pattern.
6. `losses.py::flow_matching_loss` -- similar structure to `diffusion_loss` (which is provided).
7. `dagger.py::DeterministicExpert.act` -- minor variant of `Expert.act`.
8. `dagger.py::rollout_and_relabel` -- follow the pattern in `collect_expert_data`.

That order gives you a working baseline, then a generative alternative, then an interactive correction method.
