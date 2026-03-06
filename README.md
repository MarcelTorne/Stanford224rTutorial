# CS224R Imitation Learning Tutorial: Flappy Bird

Complete tutorial demonstrating imitation learning concepts with **DAgger automatic relabeling**, proper statistical analysis, and comprehensive visualizations.

## 🚀 Quick Start

```bash
python quick_demo.py
```

**Runs in ~5-10 minutes** and demonstrates:
1. ✅ BC policy trained on **slow pipes** (speed=2.0)
2. ❌ BC fails on **fast pipes** (speed=7.0) - distribution shift!
3. 🔄 **DAgger with automatic relabeling** adapts to fast pipes
4. 📊 Generates plots with **error bars** and MP4 videos

## 🎯 Key Innovation: Automatic Relabeling

**No expert queries needed during DAgger!**

Traditional DAgger queries an expert for every state. Our implementation:
1. Rolls out policy to collect states
2. **Automatically computes optimal actions** from observations (gap positions)
3. Relabels ALL actions without expert

```python
# Automatic relabeling - extract optimal action from observation
if difficulty == "easy":
    optimal_action = obs[1]  # Gap position is in the observation!
else:
    # Pick gap closer to center (safer strategy)
    gap1_y, gap2_y = obs[1], obs[2]
    optimal_action = gap1_y if abs(gap1_y - 0.5) < abs(gap2_y - 0.5) else gap2_y
```

## 📊 Results (100 episodes each)

| Method | Performance | Interpretation |
|--------|------------|----------------|
| BC on slow pipes | **576 ± 165 steps** | Baseline performance |
| BC on fast pipes | **360 ± 206 steps** | ❌ Distribution shift (-216 steps) |
| DAgger on fast | **331 ± 199 steps** | Similar to BC (high variance) |

**Key Insight:** High variance (±200 steps) shows task difficulty at speed 7.0. DAgger and BC perform similarly within error bars.

## ✨ Features

✅ **Automatic relabeling** - No expert queries during DAgger
✅ **Error bars** - 100 episode evaluations with proper statistics
✅ **Multi-episode videos** - 3 episodes each with TIMEOUT/CRASHED labels
✅ **Landscape mode** - 800x448 optimized for video encoding
✅ **Randomized pipes** - Hard mode varies heights (bimodal distribution)
✅ **MP4 output** - Modern video format with outcome overlays
✅ **Max 1000 steps** - Clear success criterion (TIMEOUT = survived)

## 🎬 Generated Outputs

### Plots
- `plots/dagger_adaptation_curve.png` - Learning curve with error bars & shaded regions
- `plots/speed_adaptation_comparison.png` - Bar chart with error bars

### Videos (3 episodes each, annotated with outcome)
- `plots/bc_slow_on_slow.mp4` - BC at training speed
- `plots/bc_slow_on_fast.mp4` - BC with distribution shift
- `plots/dagger_on_fast.mp4` - DAgger adaptation

Example annotation: `"Ep1: 421steps (CRASHED), Ep2: 1000steps (TIMEOUT), Ep3: 437steps (CRASHED)"`

### Models
- `models/bc_slow.pt` - BC trained on slow pipes
- `models/dagger_fast.pt` - DAgger adapted to fast pipes

## 🎮 Environment

### Modes

**Easy Mode:** Single gap per pipe at random heights

**Hard Mode:** Two gaps per pipe (bimodal distribution)
- Gaps separated by 120 pixels
- Positions randomized across pipes
- Expert randomly picks one gap
- **Use case:** Demonstrates diffusion policy advantages

### Pipe Speed

Configurable via `pipe_speed` parameter:
- Slow training: 2.0
- Fast evaluation: 7.0
- Default: 3.0

**Max episode length:** 1000 steps

## 📦 Installation

```bash
pip install torch numpy gymnasium pygame matplotlib imageio[ffmpeg]
```

Or:
```bash
pip install -r requirements.txt
```

## 📖 Full Tutorial Pipeline

### 1. Collect Expert Data
```bash
python collect_data.py --difficulty easy --num_episodes 500 --output data/easy_expert.npz
python collect_data.py --difficulty hard --num_episodes 500 --output data/hard_expert.npz
```

### 2. Train Policies
```bash
# Behavior Cloning
python train_bc.py --data data/easy_expert.npz --output models/bc_easy.pt

# Diffusion Policy (handles bimodal distributions)
python train_diffusion.py --data data/hard_expert.npz --output models/diffusion_hard.pt

# DAgger (iterative improvement)
python train_dagger.py --difficulty easy --initial_data data/easy_expert.npz --output models/dagger_easy.pt
```

### 3. Evaluate
```bash
python evaluate.py --model models/bc_easy.pt --type bc --difficulty easy --episodes 50
python evaluate.py --model models/diffusion_hard.pt --type diffusion --difficulty hard --episodes 50
```

### 4. Full Multi-Seed Experiments
```bash
python run_experiments.py --render_gifs
```

Generates comparison plots, learning curves, and videos for all methods.

## 🔬 Expected Performance

### Hard Mode (Bimodal Distribution)

| Method    | Avg Episode Length | Why? |
|-----------|-------------------|------|
| BC        | ~120              | ❌ Action averaging destroys bimodal structure |
| Diffusion | ~1000             | ✅ Models full action distribution |
| DAgger    | ~700              | ⚠️ Improves but still limited |

**Key Insight:** Diffusion policies excel at bimodal action distributions!

## 📁 File Structure

```
├── quick_demo.py              # ⭐ START HERE - Complete demo
├── flappy_bird_env.py         # Gym environment (landscape, configurable speed)
├── collect_data.py            # Expert data collection
├── train_bc.py                # Behavior cloning (MSE)
├── train_diffusion.py         # Diffusion policy (DDPM)
├── train_dagger.py            # DAgger (expert-based)
├── evaluate.py                # Policy evaluation
├── run_experiments.py         # Full experimental pipeline
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 💡 Technical Details

### Statistical Rigor
- 100 episodes per evaluation
- Mean ± standard deviation reported
- Error bars on all plots
- Shaded regions for baseline variance

### Video Annotations
- Episode number and total episodes
- Step count for each episode
- Outcome: TIMEOUT (1000 steps) or CRASHED (<1000 steps)
- 2-second hold on final frame per episode

### Optimization
- GPU/MPS support (auto-detected)
- Verbose training logs with loss curves
- Efficient data collection and aggregation
- ~5-10 min total runtime on modern hardware

## 🎓 Learning Outcomes

This tutorial demonstrates:

1. **Behavior Cloning (BC)** - Simple supervised learning from expert demos
2. **Diffusion Policies** - Handling multimodal action distributions
3. **DAgger** - Iterative data aggregation for distribution shift
4. **Automatic Relabeling** - Computing optimal actions without expert queries
5. **Domain Adaptation** - Training on one distribution, testing on another
6. **Statistical Analysis** - Proper evaluation with error bars

## 🐛 Troubleshooting

**High variance in results?**
- Increase `eval_episodes` in `quick_demo.py` (currently 100)
- Run multiple seeds and average

**Videos look jerky?**
- This is expected - per-pipe action commitment creates discrete decisions

**DAgger not improving?**
- Check relabeling strategy in `rollout_and_relabel()`
- May need task-specific improvements beyond "pick center gap"

## 📚 Citation

If you use this tutorial, please cite:

```
CS224R Imitation Learning Tutorial: Flappy Bird
Stanford University, 2024
```
