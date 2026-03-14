## Setup

See [installation.md](installation.md) for instructions.

## Solution

This is the **solution** version of HW1. All TODO functions are implemented.
See [hw1_soln.md](hw1_soln.md) for the solution code snippets.

## Run the code

```
python main.py                              # full pipeline
python main.py --method bc --env easy       # BC on easy (should work)
python main.py --method bc --env hard       # BC on hard (should fail)
python main.py --method flow_matching --env hard  # flow matching (should work)
python main.py --method dagger --env hard   # DAgger (should improve)
```

## Expected Results

| Method | Easy Mode | Hard Mode |
|--------|-----------|-----------|
| BC (MSE) | ~900-1000 | ~100-200 (crashes) |
| Flow Matching | ~900-1000 | ~800-1000 |
| DAgger (5 rounds) | -- | ~600-1000 |
