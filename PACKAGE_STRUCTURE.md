# Package structure

This document describes the Python package layout in `src/cinic10`.

## High-level tree

```text
src/cinic10/                       # main package root
  __init__.py                      # package exports and public symbols
  config.py                        # typed configs and single-seed grid builder
  data.py                          # dataloaders, transforms, dataset resolving (`validate`/`valid`)
  utils.py                         # shared utility and telemetry helpers
  experiments/                     # CLI entrypoints
    __init__.py                    # experiments package marker
    run_train.py                   # single supervised training run
    run_grid_search.py             # per-seed grid workflow (24 runs per invocation)
    run_fewshot.py                 # few-shot experiment runner
    run_nas_two_stage.py           # NAS search + discrete retraining workflow
  models/                          # model construction and architectures
    __init__.py                    # model package exports
    factory.py                     # architecture selection and instantiation
    nas_cnn.py                     # Nas-derived custom CNN architecture
    convkan.py                     # ConvKAN-like layers and conversion helpers
  training/                        # supervised training stack
    __init__.py                    # training package exports
    engine.py                      # train/eval loops, progress bars, checkpoints, telemetry
    optimizer.py                   # optimizer and scheduler builders
  fewshot/                         # episodic few-shot stack
    __init__.py                    # few-shot package exports
    protonet.py                    # Prototypical Network training/evaluation + telemetry
```

## Runtime flow

1. A CLI in `experiments/` parses arguments and builds config.
2. `data.py` resolves dataset root and creates dataloaders (or episodic sampling in `fewshot/`).
3. `models/factory.py` builds the selected architecture.
4. `training/engine.py` or `fewshot/protonet.py` runs training/evaluation.
5. `utils.py` + training modules persist metrics, checkpoints, and optimized weights.

## Notes

- Multi-seed orchestration lives in `EXPERIMENT_PLAN.md` (external seed loops).
- `run_grid_search.py` is intentionally single-seed; run once per seed.
