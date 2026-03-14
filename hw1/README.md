## Setup

See [installation.md](installation.md) for instructions.

## Complete the code

Fill in sections marked with `TODO`. In particular, see
 - [networks.py](networks.py) — `BCPolicy`
 - [losses.py](losses.py) — `bc_loss`, `flow_matching_loss`
 - [expert.py](expert.py) — `Expert.act`
 - [dagger.py](dagger.py) — `DeterministicExpert.act`, `rollout_and_relabel`

Look for sections marked with `HW1` in other files to see how your edits will be used.
Some other files that you may find relevant
 - [main.py](main.py) — training pipeline and evaluation (read-only)
 - [visualization.py](visualization.py) — evaluation and video utilities (read-only)

See the homework pdf for more details.

## Run the code

Tip: Use specific `--method` and `--env` flags to run one part at a time instead of the full pipeline.

### Part 1: Behavior Cloning on Easy Mode

Implement `BCPolicy`, `bc_loss`, and `Expert.act`, then verify BC works on easy mode:

```
python main.py --method bc --env easy
```

You should see ~900-1000 avg episode length.

### Part 2: BC Fails on Hard Mode

Run the same BC policy on hard mode and observe it crashing (~100-200 avg steps):

```
python main.py --method bc --env hard
```

### Part 3: Flow Matching on Hard Mode

Implement `flow_matching_loss`, then run flow matching on hard mode:

```
python main.py --method flow_matching --env hard
```

You should see ~800-1000 avg episode length.

### Part 4: DAgger on Hard Mode

Implement `DeterministicExpert.act` and `rollout_and_relabel`, then run DAgger:

```
python main.py --method dagger --env hard
```

You should see progressive improvement from ~100-200 to ~600-1000 avg steps.

### Full Pipeline

```
python main.py
```

Outputs (plots, videos, model checkpoints) are saved to timestamped subdirectories under `plots/` and `models/`.

## Environment Details

**Observation** (4-D): `[dist_to_pipe, gap1_y, gap2_y, bird_y]` (normalised 0-1).

**Action**: target y position in [0, 1]. The environment uses a PD controller to convert the target into thrust.

**Easy mode**: one gap per pipe (unimodal). **Hard mode**: two gaps per pipe (bimodal expert demonstrations).

**Max episode length**: 1000 steps.

## Expected Results

| Method | Easy Mode (avg steps) | Hard Mode (avg steps) |
|--------|----------------------|----------------------|
| BC (MSE) | ~900-1000 | ~100-200 (crashes) |
| Flow Matching | ~900-1000 | ~800-1000 |
| DAgger (5 rounds) | -- | ~600-1000 |
