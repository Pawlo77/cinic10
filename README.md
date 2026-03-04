# 🖼️ CINIC-10 Experiment Suite

Framework for CINIC-10 supervised and few-shot experiments with reproducible runs, resumable checkpoints, and per-epoch resource telemetry.

## Docs

- [EXPERIMENT_PLAN.md](./EXPERIMENT_PLAN.md)
- [PACKAGE_STRUCTURE.md](./PACKAGE_STRUCTURE.md)

## Quick Start

```bash
make install
make help
```

Set environment once:

```bash
export DATA_ROOT=/path/to/cinic10
export DEVICE=mps
```

If `DATA_ROOT` is missing, auto-download from Kaggle is used. To disable auto-download:

```bash
export CINIC10_AUTO_DOWNLOAD=0
```

## Core Commands

Their descriptions and hyperparameter options are in `EXPERIMENT_PLAN.md`. Refer to `make help` for usage.

```bash
make train
make train-mixup
make train-cutmix
make train-reduced
make grid
make grid-resume
make nas-two-stage
make nas-two-stage-resume
make fewshot
make fewshot-resume
```

All targets accept overrides (example):

```bash
make train DATA_ROOT=data OUTPUT_DIR=outputs/run1 ARCH=resnet18 SEED=42 DEVICE=cpu
```

## Experiment Workflow

Full, seed-aware execution order is documented in [EXPERIMENT_PLAN.md](./EXPERIMENT_PLAN.md).

## Artifacts

- `metrics.json`, `test_metrics.json`
- `epoch_resource_stats.json` (supervised per-epoch timing/memory)
- `grid_results.csv`, `grid_results.json`
- `architecture.json`, `two_stage_summary.json`
- `fewshot_metrics.json`, `fewshot_episode_resource_stats.json`

## Package Layout

See [PACKAGE_STRUCTURE.md](./PACKAGE_STRUCTURE.md).
