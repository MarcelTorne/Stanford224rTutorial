# CS224R Imitation Learning Homework: Flappy Bird

Imitation learning on a Flappy Bird environment with **action chunking**.
You will implement core components -- neural networks, loss functions, an expert
policy, and DAgger relabeling -- then run the full pipeline to compare
Behavior Cloning, Diffusion Policy, Gaussian BC, and DAgger.

## Setup

```bash
pip install -r requirements.txt
```

## File Structure

```
flappy_bird_env.py   # Gymnasium environment (provided, do not modify)
networks.py          # Neural network architectures         [TODO: BCPolicy, GaussianBCPolicy]
losses.py            # Loss functions                       [TODO: all three]
expert.py            # Expert policy + data collection      [TODO: Expert.act]
dagger.py            # DAgger relabeling                    [TODO: DeterministicExpert.act, rollout_and_relabel]
visualization.py     # Evaluation & video utils             (provided, do not modify)
main.py              # Training helpers & full pipeline      (provided, do not modify)
assets/              # Sprites for rendering
data/                # Pre-collected expert data (.npz)
requirements.txt     # Dependencies
```

## Your Tasks

Every function you need to implement is marked with
`raise NotImplementedError("TODO: ...")`. Search for `TODO` across the
codebase to find them all.

### 1. Neural Networks (`networks.py`)

- **`BCPolicy`**: simple 3-layer MLP (state -> action) with Sigmoid output.
- **`GaussianBCPolicy`**: shared backbone with separate mean and log-variance
  heads. Implement `forward`, `sample`, and `deterministic`.

### 2. Loss Functions (`losses.py`)

- **`bc_loss`**: MSE between predicted and target actions.
- **`gaussian_nll_loss`**: Gaussian negative log-likelihood using predicted
  mean and log-variance.
- **`diffusion_loss`**: MSE between predicted and actual noise (denoising
  objective).

### 3. Expert Policy (`expert.py`)

- **`Expert.act`**: easy mode targets the single gap; hard mode hovers at the
  midpoint of two gaps and randomly commits to one when close. Uses EMA
  smoothing.

### 4. DAgger (`dagger.py`)

- **`DeterministicExpert.act`**: same as the stochastic expert but *always*
  picks the upper gap (gap1). This is the key to resolving bimodal ambiguity.
- **`rollout_and_relabel`**: roll out the current policy in the environment,
  relabel every visited state with the deterministic expert, and window the
  results into action-chunk training pairs.

## Running

Once all TODOs are filled in:

```bash
python main.py
```

The pipeline runs three parts:

1. **Hard mode** -- trains BC, Diffusion, and Gaussian BC on bimodal expert
   data. BC averages the two modes and crashes; Diffusion models the full
   distribution and succeeds.
2. **DAgger** -- starting from the pretrained BC, iteratively relabels with a
   deterministic expert to resolve the bimodal ambiguity.
3. **Easy mode** -- sanity check: both BC and Diffusion work on unimodal data.

Outputs (plots, videos, model checkpoints) are saved to timestamped
subdirectories under `plots/` and `models/`.

## Environment Details

**Observation** (4-D): `[dist_to_pipe, gap1_y, gap2_y, bird_y]` (normalised).

**Action**: target y position in [0, 1]. The environment uses a PD controller
to convert the target into thrust.

**Easy mode**: one gap per pipe. **Hard mode**: two gaps per pipe (bimodal
expert demonstrations).

**Max episode length**: 1000 steps.

## Expected Results

| Method | Hard Mode (avg steps) | Easy Mode (avg steps) |
|--------|----------------------|----------------------|
| BC (MSE) | ~100-200 (crashes) | ~900-1000 |
| Gaussian BC (det) | ~100-200 (crashes) | -- |
| Diffusion (DDPM) | ~800-1000 | ~900-1000 |
| DAgger (5 rounds) | ~600-1000 | -- |
