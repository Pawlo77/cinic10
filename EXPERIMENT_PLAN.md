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

Search for the ideal number of epochs:

```bash
for SEED in ${=SEEDS}; do
  export CINIC10_LOG_FILE_NAME="grid_epochs_mobilenet_seed_${SEED}.log"
  echo "Starting epoch grid for seed ${SEED} at $(date)"
  make grid \
    OUTPUT_ROOT=outputs/01_grid__epochs_mobilenet/seed_${SEED} \
    EXTRA_ARGS="--stop-after 0 --epochs 100" \
    SEED=$SEED
  echo "Completed epoch grid for seed ${SEED} at $(date)"
done
```

Reusable helper (keeps the plan short):

```bash
run_seeded_train() {
  local OUT_PREFIX="$1"
  local ARCH_NAME="$2"
  local EXTRA="$3"
  for SEED in ${=SEEDS}; do
    export CINIC10_LOG_DIR="logs/${OUT_PREFIX}"
    export CINIC10_LOG_FILE_NAME="train_${ARCH_NAME}_seed_${SEED}.log"
    make train \
      OUTPUT_DIR="outputs/${OUT_PREFIX}/seed_${SEED}" \
      ARCH=$ARCH_NAME \
      OPTIMIZER=$BEST_OPTIMIZER \
      BATCH_SIZE=$BEST_BATCH_SIZE \
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
for SEED in ${=SEEDS}; do
  echo "Starting epoch grid for seed ${SEED} at $(date)"
  export CINIC10_LOG_FILE_NAME="grid_mobilenet_seed_${SEED}.log"
  make grid \
    OUTPUT_ROOT=outputs/01_grid_mobilenet/seed_${SEED} \
    SEED=$SEED

done
```

If interrupted:

```bash
for SEED in ${=SEEDS}; do
  echo "Starting epoch grid for seed ${SEED} at $(date)"
  export CINIC10_LOG_FILE_NAME="grid_mobilenet_seed_${SEED}.log"
  make grid-resume \
    OUTPUT_ROOT=outputs/01_grid_mobilenet/seed_${SEED} \
    SEED=$SEED
  echo "Completed epoch grid for seed ${SEED} at $(date)"
done
```

Review `outputs/01_grid_mobilenet/seed_*/grid_results.csv` and identify best hyperparameters (optimizer, batch size, epochs, dropout, weight decay).

Set selected hyperparameters for the next stages (update these values based on grid results):

```bash
export BEST_OPTIMIZER=adamw
export BEST_BATCH_SIZE=128
export BEST_EPOCHS=60
export BEST_DROPOUT=0.5
export WEIGHT_DECAY=0.0001
```

---

## 2) Augmentation ablation on MobileNetV3 Small

Run all five variants per seed:

```bash
for SEED in ${=SEEDS}; do
  echo "Starting epoch grid for seed ${SEED} at $(date)"
  export CINIC10_LOG_FILE_NAME="aug_mobilenet_seed_${SEED}.log"

  make train-no-aug OUTPUT_DIR=outputs/02_aug/mobilenet_none/seed_${SEED} ARCH=mobilenet_v3_small OPTIMIZER=$BEST_OPTIMIZER BATCH_SIZE=$BEST_BATCH_SIZE EPOCHS=$BEST_EPOCHS SEED=$SEED EXTRA_ARGS="--weight-decay $WEIGHT_DECAY --dropout $BEST_DROPOUT" && \

  make train OUTPUT_DIR=outputs/02_aug/mobilenet_standard/seed_${SEED} ARCH=mobilenet_v3_small OPTIMIZER=$BEST_OPTIMIZER BATCH_SIZE=$BEST_BATCH_SIZE EPOCHS=$BEST_EPOCHS SEED=$SEED EXTRA_ARGS="--weight-decay $WEIGHT_DECAY --dropout $BEST_DROPOUT" && \

  make train-mixup OUTPUT_DIR=outputs/02_aug/mobilenet_standard_mixup/seed_${SEED} ARCH=mobilenet_v3_small OPTIMIZER=$BEST_OPTIMIZER BATCH_SIZE=$BEST_BATCH_SIZE EPOCHS=$BEST_EPOCHS SEED=$SEED EXTRA_ARGS="--weight-decay $WEIGHT_DECAY --dropout $BEST_DROPOUT" && \

  make train-cutmix OUTPUT_DIR=outputs/02_aug/mobilenet_standard_cutmix/seed_${SEED} ARCH=mobilenet_v3_small OPTIMIZER=$BEST_OPTIMIZER BATCH_SIZE=$BEST_BATCH_SIZE EPOCHS=$BEST_EPOCHS SEED=$SEED EXTRA_ARGS="--weight-decay $WEIGHT_DECAY --dropout $BEST_DROPOUT" && \

  make train-autoaugment OUTPUT_DIR=outputs/02_aug/mobilenet_autoaugment/seed_${SEED} ARCH=mobilenet_v3_small OPTIMIZER=$BEST_OPTIMIZER BATCH_SIZE=$BEST_BATCH_SIZE EPOCHS=$BEST_EPOCHS SEED=$SEED EXTRA_ARGS="--weight-decay $WEIGHT_DECAY --dropout $BEST_DROPOUT"

done
```

Pick best augmentation and set `BEST_AUG`.

```bash
export BEST_AUG=standard
```

---

## 3) Train other architectures with best hyperparams + best augmentation

With early stopping—stops training if validation loss doesn't improve by ≥0.01 for 10 consecutive epochs:

```bash
run_seeded_train_with_best_aug() {
  local OUT_PREFIX="$1"
  local ARCH_NAME="$2"
  local EXTRA="$3"
  local TARGET=""

  case "$BEST_AUG" in
    none) TARGET="train-no-aug" ;;
    standard) TARGET="train" ;;
    standard_mixup) TARGET="train-mixup" ;;
    standard_cutmix) TARGET="train-cutmix" ;;
    autoaugment) TARGET="train-autoaugment" ;;
    *)
      echo "Unsupported BEST_AUG='$BEST_AUG'. Use one of: none, standard, standard_mixup, standard_cutmix, autoaugment"
      return 1
      ;;
  esac

  for SEED in ${=SEEDS}; do
    export CINIC10_LOG_DIR="logs/${OUT_PREFIX}"
    export CINIC10_LOG_FILE_NAME="train_${ARCH_NAME}_seed_${SEED}.log"
    make $TARGET \
      OUTPUT_DIR="outputs/${OUT_PREFIX}/seed_${SEED}" \
      ARCH=$ARCH_NAME \
      OPTIMIZER=$BEST_OPTIMIZER \
      BATCH_SIZE=$BEST_BATCH_SIZE \
      EPOCHS=$BEST_EPOCHS \
      SEED=$SEED \
      EXTRA_ARGS="--weight-decay $WEIGHT_DECAY --dropout $BEST_DROPOUT --early-stopping $EXTRA"
  done
}

run_seeded_train_with_best_aug 03_models/squeezenet squeezenet1_0 "" && \
run_seeded_train_with_best_aug 03_models/resnet18_finetune resnet18 "--pretrained" && \
run_seeded_train_with_best_aug 03_models/densenet121_finetune densenet121 "--pretrained" && \
# run_seeded_train_with_best_aug 03_models/convkan_mobilenet_v3_small convkan_mobilenet_v3_small "" && \
# run_seeded_train_with_best_aug 03_models/convkan_squeezenet1_0 convkan_squeezenet1_0 "" && \

for SEED in ${=SEEDS}; do
  export CINIC10_LOG_DIR="logs/03_models/nas_two_stage"
  export CINIC10_LOG_FILE_NAME="nas_two_stage_seed_${SEED}.log"
  make nas-two-stage OUTPUT_ROOT=outputs/03_models/nas_two_stage/seed_${SEED} SEED=$SEED OPTIMIZER=$BEST_OPTIMIZER BATCH_SIZE=$BEST_BATCH_SIZE EPOCHS_SEARCH=$BEST_EPOCHS EPOCHS_RETRAIN=$BEST_EPOCHS EXTRA_ARGS="--augmentation $BEST_AUG --weight-decay $WEIGHT_DECAY --dropout $BEST_DROPOUT" && \
done
```

Resume NAS if needed:

```bash
for SEED in ${=SEEDS}; do
  export CINIC10_LOG_DIR="logs/03_models/nas_two_stage"
  export CINIC10_LOG_FILE_NAME="nas_two_stage_resume_seed_${SEED}.log"
  make nas-two-stage-resume OUTPUT_ROOT=outputs/03_models/nas_two_stage/seed_${SEED} SEED=$SEED OPTIMIZER=$BEST_OPTIMIZER BATCH_SIZE=$BEST_BATCH_SIZE EPOCHS_SEARCH=$BEST_EPOCHS EPOCHS_RETRAIN=$BEST_EPOCHS EXTRA_ARGS="--augmentation $BEST_AUG --weight-decay $WEIGHT_DECAY --dropout $BEST_DROPOUT"
done
```

Train NAS-derived ConvKAN architecture from scratch (loads searched architecture for each seed from `outputs/03_models/nas_two_stage`):

```bash
for SEED in ${=SEEDS}; do
  export CINIC10_LOG_DIR="logs/03_models/nas_convkan"
  export CINIC10_LOG_FILE_NAME="nas_convkan_seed_${SEED}.log"
  make nas-convkan \
    OUTPUT_DIR=outputs/03_models/nas_convkan/seed_${SEED} \
    NAS_OUTPUT_ROOT=outputs/03_models/nas_two_stage \
    CONVKAN_MIN_KERNEL_SIZE=3 \
    SEED=$SEED \
    OPTIMIZER=$BEST_OPTIMIZER \
    BATCH_SIZE=$BEST_BATCH_SIZE \
    EPOCHS=$BEST_EPOCHS \
    EXTRA_ARGS="--augmentation $BEST_AUG --weight-decay $WEIGHT_DECAY --dropout $BEST_DROPOUT --early-stopping" && \
done
```

---

Select the best architecture for the next stage and set `$FINAL_ARCH`

```bash
export FINAL_ARCH=densenet121
export FINAL_AUG_EXTRA_ARGS="--augmentation $BEST_AUG --weight-decay $WEIGHT_DECAY --dropout $BEST_DROPOUT --early-stopping"
```

## 4) Final stage with best model configuration

```bash
for SEED in ${=SEEDS}; do
  export CINIC10_LOG_DIR="logs/04_final/dataset_reduction_5pct"
  export CINIC10_LOG_FILE_NAME="train_${FINAL_ARCH}_seed_${SEED}.log"
  make train-reduced OUTPUT_DIR=outputs/04_final/dataset_reduction_5pct/seed_${SEED} ARCH=$FINAL_ARCH OPTIMIZER=$BEST_OPTIMIZER BATCH_SIZE=$BEST_BATCH_SIZE EPOCHS=$BEST_EPOCHS SEED=$SEED TRAIN_FRACTION=0.05 EXTRA_ARGS="$FINAL_AUG_EXTRA_ARGS"

  export CINIC10_LOG_DIR="logs/04_final/fewshot_protonet"
  export CINIC10_LOG_FILE_NAME="fewshot_seed_${SEED}.log"
  make fewshot OUTPUT_DIR=outputs/04_final/fewshot_protonet/seed_${SEED} SEED=$SEED WAYS=5 SHOTS=5 QUERIES=15 EPISODES=2000 EVAL_EPISODES=400
done
```

Resume few-shot if needed:

```bash
for SEED in ${=SEEDS}; do
  export CINIC10_LOG_DIR="logs/04_final/fewshot_protonet"
  export CINIC10_LOG_FILE_NAME="fewshot_resume_seed_${SEED}.log"
  make fewshot-resume OUTPUT_DIR=outputs/04_final/fewshot_protonet/seed_${SEED} SEED=$SEED
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
