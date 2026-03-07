# CINIC-10 Experiment Execution Plan

## 0) Common setup

```bash
make install
```

Set shared variables once in your shell (or in `.env`):

```bash
export DATA_ROOT=/path/to/cinic10
export DEVICE=mps
export SEEDS="0 42 3407"
```

All stages run per seed and store results in `.../seed_<seed>`.

Set selected hyperparameters after Stage 1:

```bash
export BEST_OPTIMIZER=adamw
export BEST_BATCH_SIZE=128
export BEST_LR=3e-4
export BEST_EPOCHS=30
export BEST_AUG_EXTRA_ARGS=""
export FINAL_ARCH=resnet18
export FINAL_AUG_EXTRA_ARGS="$BEST_AUG_EXTRA_ARGS"
```

Reusable helper (keeps the plan short):

```bash
run_seeded_train() {
  local OUT_PREFIX="$1"
  local ARCH_NAME="$2"
  local EXTRA="$3"
  for SEED in $SEEDS; do
    make train \
      DATA_ROOT=$DATA_ROOT \
      OUTPUT_DIR=${OUT_PREFIX}/seed_${SEED} \
      DEVICE=$DEVICE \
      ARCH=$ARCH_NAME \
      OPTIMIZER=$BEST_OPTIMIZER \
      BATCH_SIZE=$BEST_BATCH_SIZE \
      LR=$BEST_LR \
      EPOCHS=$BEST_EPOCHS \
      SEED=$SEED \
      EXTRA_ARGS="$EXTRA"
  done
}
```

---

## 1) Hyperparameter grid on MobileNetV3 Small

Run the per-seed grid (24 runs per seed):

```bash
for SEED in $SEEDS; do
  make grid \
    DATA_ROOT=$DATA_ROOT \
    OUTPUT_ROOT=outputs/01_grid_mobilenet/seed_${SEED} \
    DEVICE=$DEVICE \
    SEED=$SEED
done
```

If interrupted:

```bash
for SEED in $SEEDS; do
  make grid-resume \
    DATA_ROOT=$DATA_ROOT \
    OUTPUT_ROOT=outputs/01_grid_mobilenet/seed_${SEED} \
    DEVICE=$DEVICE \
    SEED=$SEED
done
```

Review `outputs/01_grid_mobilenet/seed_*/grid_results.csv` and set `BEST_*` values.

---

## 2) Augmentation ablation on MobileNetV3 Small

Run all five variants per seed:

```bash
run_seeded_train outputs/02_aug/mobilenet_none mobilenet_v3_small "--augmentation none"
run_seeded_train outputs/02_aug/mobilenet_standard mobilenet_v3_small "--augmentation standard"
for SEED in $SEEDS; do
  make train-mixup DATA_ROOT=$DATA_ROOT OUTPUT_DIR=outputs/02_aug/mobilenet_standard_mixup/seed_${SEED} DEVICE=$DEVICE ARCH=mobilenet_v3_small OPTIMIZER=$BEST_OPTIMIZER BATCH_SIZE=$BEST_BATCH_SIZE LR=$BEST_LR EPOCHS=$BEST_EPOCHS SEED=$SEED
  make train-cutmix DATA_ROOT=$DATA_ROOT OUTPUT_DIR=outputs/02_aug/mobilenet_standard_cutmix/seed_${SEED} DEVICE=$DEVICE ARCH=mobilenet_v3_small OPTIMIZER=$BEST_OPTIMIZER BATCH_SIZE=$BEST_BATCH_SIZE LR=$BEST_LR EPOCHS=$BEST_EPOCHS SEED=$SEED
done
run_seeded_train outputs/02_aug/mobilenet_autoaugment mobilenet_v3_small "--augmentation autoaugment"
```

Pick best augmentation and set `BEST_AUG_EXTRA_ARGS`.

---

## 3) Train other architectures with best hyperparams + best augmentation

```bash
run_seeded_train outputs/03_models/squeezenet squeezenet1_0 "$BEST_AUG_EXTRA_ARGS"
run_seeded_train outputs/03_models/resnet18_finetune resnet18 "--pretrained $BEST_AUG_EXTRA_ARGS"
run_seeded_train outputs/03_models/densenet121_finetune densenet121 "--pretrained $BEST_AUG_EXTRA_ARGS"
run_seeded_train outputs/03_models/convkan_mobilenet_v3_small convkan_mobilenet_v3_small "$BEST_AUG_EXTRA_ARGS"
run_seeded_train outputs/03_models/convkan_squeezenet1_0 convkan_squeezenet1_0 "$BEST_AUG_EXTRA_ARGS"

for SEED in $SEEDS; do
  make nas-two-stage DATA_ROOT=$DATA_ROOT OUTPUT_ROOT=outputs/03_models/nas_two_stage/seed_${SEED} DEVICE=$DEVICE SEED=$SEED BATCH_SIZE=$BEST_BATCH_SIZE LR=$BEST_LR EPOCHS_SEARCH=$BEST_EPOCHS EPOCHS_RETRAIN=$BEST_EPOCHS
done
```

Resume NAS if needed:

```bash
for SEED in $SEEDS; do
  make nas-two-stage-resume DATA_ROOT=$DATA_ROOT OUTPUT_ROOT=outputs/03_models/nas_two_stage/seed_${SEED} DEVICE=$DEVICE SEED=$SEED BATCH_SIZE=$BEST_BATCH_SIZE LR=$BEST_LR EPOCHS_SEARCH=$BEST_EPOCHS EPOCHS_RETRAIN=$BEST_EPOCHS
done
```

---

## 4) Final stage with best model configuration

```bash
for SEED in $SEEDS; do
  make train-reduced DATA_ROOT=$DATA_ROOT OUTPUT_DIR=outputs/04_final/dataset_reduction_5pct/seed_${SEED} DEVICE=$DEVICE ARCH=$FINAL_ARCH OPTIMIZER=$BEST_OPTIMIZER BATCH_SIZE=$BEST_BATCH_SIZE LR=$BEST_LR EPOCHS=$BEST_EPOCHS SEED=$SEED TRAIN_FRACTION=0.05 EXTRA_ARGS="$FINAL_AUG_EXTRA_ARGS"
  make fewshot DATA_ROOT=$DATA_ROOT OUTPUT_DIR=outputs/04_final/fewshot_protonet/seed_${SEED} DEVICE=$DEVICE SEED=$SEED WAYS=5 SHOTS=5 QUERIES=15 EPISODES=2000 EVAL_EPISODES=400
done
```

Resume few-shot if needed:

```bash
for SEED in $SEEDS; do
  make fewshot-resume DATA_ROOT=$DATA_ROOT OUTPUT_DIR=outputs/04_final/fewshot_protonet/seed_${SEED} DEVICE=$DEVICE SEED=$SEED
done
```

---

## 5) Minimal result checklist

After each stage, collect:
- `metrics.json`
- `test_metrics.json` (for supervised runs)
- `epoch_resource_stats.json` (per-epoch wall time, CPU time, RAM, accelerator memory)
- `grid_results.csv` (per-seed grid stage)
- `architecture.json` and `two_stage_summary.json` (NAS stage)
- `fewshot_metrics.json` (few-shot stage)
- `fewshot_episode_resource_stats.json` (per-episode wall time, CPU time, RAM, accelerator memory)

This gives a full trace from broad search to final low-data and few-shot evaluation with 3-seed statistics.
