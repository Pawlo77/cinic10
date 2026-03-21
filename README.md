# 🖼️ CINIC-10 Experiment Suite

Framework for CINIC-10 supervised and few-shot experiments with reproducible runs, resumable checkpoints, and per-epoch resource telemetry.

## Features

- **Multiple architectures**: MobileNetV3 Small, SqueezeNet, ResNet18, DenseNet121, ConvKAN variants, NAS-derived CNN
- **Augmentation strategies**: None, Standard, MixUp, CutMix, AutoAugment
- **Early stopping**: Optional validation loss–based stopping (improves by <0.01 for 10 epochs)
- **Few-shot learning**: Prototypical Networks with episodic training
- **Hyperparameter grid search**: Automated 24-run grid on MobileNetV3
- **Two-stage NAS**: Architecture search followed by discrete retraining
- **Resource telemetry**: Per-epoch/episode wall time, CPU time, and memory tracking
- **Resumable checkpoints**: All experiments support interruption and resume

## Docs

- [EXPERIMENT_PLAN.md](./EXPERIMENT_PLAN.md) — Stage-by-stage workflow with seed loops
- [PACKAGE_STRUCTURE.md](./PACKAGE_STRUCTURE.md) — Python package layout and runtime flow

## Quick Start

```bash
make install
make help
```

Set environment in `.env` file (home directory or workspace root). Variables configurable via Makefile (see top of file):

```bash
DATA_ROOT=/path/to/cinic10
DEVICE=mps              # or cuda, cpu
SEEDS="0 42 3407"       # for multi-seed experiments
```

## Core Commands

Refer to `make help` for full descriptions. All targets accept variable overrides.

**Supervised Training:**
```bash
make train              # Standard augmentation
make train-no-aug       # No augmentation
make train-mixup        # Standard + MixUp
make train-cutmix       # Standard + CutMix
make train-autoaugment  # AutoAugment
make train-reduced      # Reduced training data (supports --early-stopping flag)
```

Optional flags:
```bash
# Use --early-stopping to stop training if val loss doesn't improve by ≥0.01 for 10 epochs
make train EXTRA_ARGS="--early-stopping"
```

**Grid & NAS:**
```bash
make grid               # 24-run hyperparameter grid (MobileNetV3)
make grid-resume        # Resume/skip completed grid runs
make nas-two-stage      # Two-stage architecture search
make nas-two-stage-resume
```

**Few-Shot Learning:**
```bash
make fewshot            # Prototypical Network
```
See [EXPERIMENT_PLAN.md](./EXPERIMENT_PLAN.md) for the complete multi-stage, multi-seed execution order:

1. **Hyperparameter grid** on MobileNetV3 Small (per seed)
2. **Augmentation ablation** (5 variants per seed)
3. **Architecture comparison** with best hyperparameters
4. **Final evaluation** with data reduction and few-shot learning

## Output Artifacts

**Per-run:**
- `metrics.json` — Training/validation metrics
- `test_metrics.json` — Final test set evaluation
- `epoch_resource_stats.json` — Per-epoch wall/CPU time and memory
- `last.ckpt` — Resumable checkpoint (optimizer + RNG state)
- `model.safetensors` — Final weights (optimized format)

**Grid search:**
- `grid_results.csv`, `grid_results.json` — All 24 runs with hyperparameters

**NAS:**
- `architecture.json` — Searched architecture definition
- `two_stage_summary.json` — Search and retrain metrics

**Few-shot:**
- `fewshot_metrics.json` — Episode-averaged accuracy
- `fewshot_episode_resource_stats.json` — Per-episode resource usage

## Package Structure

```
src/cinic10/
  config.py          # Typed configs and grid builder
  data.py            # Dataloaders and transforms
  utils.py           # Telemetry and checkpoint helpers
  experiments/       # CLI entrypoints (run_train, run_grid_search, run_fewshot, run_nas_two_stage)
  models/            # Architecture factory, NAS CNN, ConvKAN layers
  training/          # Supervised engine, optimizer builders
  fewshot/           # Prototypical Network episodic training
```

See [PACKAGE_STRUCTURE.md](./PACKAGE_STRUCTURE.md) for detailed layout and runtime flow

- `metrics.json`, `test_metrics.json`
- `epoch_resource_stats.json` (supervised per-epoch timing/memory)
- `grid_results.csv`, `grid_results.json`
- `architecture.json`, `two_stage_summary.json`
- `fewshot_metrics.json`, `fewshot_episode_resource_stats.json`

## Package Layout

See [PACKAGE_STRUCTURE.md](./PACKAGE_STRUCTURE.md).
