ifneq ("$(wildcard .env)","")
	include .env
	export
endif

UV ?= uv run
PYTHONPATH ?= src
DATA_ROOT ?= /path/to/cinic10
OUTPUT_DIR ?= outputs/train_baseline
OUTPUT_ROOT ?= outputs/grid
DEVICE ?= mps
ARCH ?= mobilenet_v3_small
EPOCHS ?= 30
BATCH_SIZE ?= 128
OPTIMIZER ?= adamw
LR ?= 3e-4
TRAIN_FRACTION ?= 1.0
SEED ?= 42
WAYS ?= 5
SHOTS ?= 5
QUERIES ?= 15
EPISODES ?= 2000
EVAL_EPISODES ?= 400
EPOCHS_SEARCH ?= 30
EPOCHS_RETRAIN ?= 30
EXTRA_ARGS ?=

.PHONY: help install clean test train train-no-aug train-mixup train-cutmix train-autoaugment train-reduced fewshot fewshot-resume grid grid-resume nas-two-stage nas-two-stage-resume

help:
	@echo "Available targets:"
	@echo "  make install                - Install dependencies and hooks"
	@echo "  make test                   - Run tests"
	@echo "  make train                  - Supervised training with standard augmentation"
	@echo "  make train-no-aug           - Supervised training with no augmentation"
	@echo "  make train-mixup            - Supervised training with MixUp"
	@echo "  make train-cutmix           - Supervised training with CutMix"
	@echo "  make train-autoaugment      - Supervised training with AutoAugment"
	@echo "  make train-reduced          - Supervised training with reduced data"
	@echo "  make fewshot                - Few-shot Prototypical Network"
	@echo "  make fewshot-resume         - Resume few-shot training"
	@echo "  make grid                   - MobileNet hyperparameter grid"
	@echo "  make grid-resume            - Resume/skip completed grid runs"
	@echo "  make nas-two-stage          - Search then retrain discrete NAS"
	@echo "  make nas-two-stage-resume   - Resume two-stage NAS"
	@echo ""
	@echo "Override vars, e.g.:"
	@echo "  make train DATA_ROOT=./data/cinic10 OUTPUT_DIR=outputs/run1 DEVICE=cpu"

# install dependencies and pre-commit hooks
install:
	uv sync --all-groups
	uv run pre-commit install

# clean up virtual environment and lockfile
clean:
	rm -rf .venv
	rm -rf uv.lock

# run tests with pytest
test::
	uv run pytest -v

# pre-commit checks (linting, formatting, type checking)
pre-commit-all:
	uv run pre-commit run --all-files

# pre-commit checks on changed files only
pre-commit:
	uv run pre-commit run

# supervised training with configurable options (see vars above)
train:
	PYTHONPATH=$(PYTHONPATH) $(UV) python -m cinic10.experiments.run_train \
		--data-root $(DATA_ROOT) \
		--output-dir $(OUTPUT_DIR) \
		--architecture $(ARCH) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--optimizer $(OPTIMIZER) \
		--learning-rate $(LR) \
		--train-fraction $(TRAIN_FRACTION) \
		--seed $(SEED) \
		--device $(DEVICE) \
		--augmentation standard $(EXTRA_ARGS)

# train with no image or batch-level augmentation
train-no-aug:
	PYTHONPATH=$(PYTHONPATH) $(UV) python -m cinic10.experiments.run_train \
		--data-root $(DATA_ROOT) \
		--output-dir $(OUTPUT_DIR) \
		--architecture $(ARCH) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--optimizer $(OPTIMIZER) \
		--learning-rate $(LR) \
		--seed $(SEED) \
		--device $(DEVICE) $(EXTRA_ARGS) \
		--augmentation none

# train with standard + MixUp augmentation
train-mixup:
	PYTHONPATH=$(PYTHONPATH) $(UV) python -m cinic10.experiments.run_train \
		--data-root $(DATA_ROOT) \
		--output-dir $(OUTPUT_DIR) \
		--architecture $(ARCH) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--optimizer $(OPTIMIZER) \
		--learning-rate $(LR) \
		--seed $(SEED) \
		--device $(DEVICE) $(EXTRA_ARGS) \
		--augmentation standard_mixup

# train with standard + CutMix augmentation
train-cutmix:
	PYTHONPATH=$(PYTHONPATH) $(UV) python -m cinic10.experiments.run_train \
		--data-root $(DATA_ROOT) \
		--output-dir $(OUTPUT_DIR) \
		--architecture $(ARCH) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--optimizer $(OPTIMIZER) \
		--learning-rate $(LR) \
		--seed $(SEED) \
		--device $(DEVICE) $(EXTRA_ARGS) \
		--augmentation standard_cutmix

# train with AutoAugment image-level augmentation
train-autoaugment:
	PYTHONPATH=$(PYTHONPATH) $(UV) python -m cinic10.experiments.run_train \
		--data-root $(DATA_ROOT) \
		--output-dir $(OUTPUT_DIR) \
		--architecture $(ARCH) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--optimizer $(OPTIMIZER) \
		--learning-rate $(LR) \
		--seed $(SEED) \
		--device $(DEVICE) $(EXTRA_ARGS) \
		--augmentation autoaugment

# train with reduced training data (train fraction < 1.0)
train-reduced:
	PYTHONPATH=$(PYTHONPATH) $(UV) python -m cinic10.experiments.run_train \
		--data-root $(DATA_ROOT) \
		--output-dir $(OUTPUT_DIR) \
		--architecture $(ARCH) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--optimizer $(OPTIMIZER) \
		--learning-rate $(LR) \
		--seed $(SEED) \
		--device $(DEVICE) \
		--train-fraction $(TRAIN_FRACTION) $(EXTRA_ARGS)

# few-shot Prototypical Network training and evaluation
fewshot:
	PYTHONPATH=$(PYTHONPATH) $(UV) python -m cinic10.experiments.run_fewshot \
		--data-root $(DATA_ROOT) \
		--output-dir $(OUTPUT_DIR) \
		--seed $(SEED) \
		--ways $(WAYS) \
		--shots $(SHOTS) \
		--queries $(QUERIES) \
		--episodes $(EPISODES) \
		--eval-episodes $(EVAL_EPISODES) \
		--device $(DEVICE) $(EXTRA_ARGS)

# resume few-shot training from last checkpoint in output dir
fewshot-resume:
	PYTHONPATH=$(PYTHONPATH) $(UV) python -m cinic10.experiments.run_fewshot \
		--data-root $(DATA_ROOT) \
		--output-dir $(OUTPUT_DIR) \
		--seed $(SEED) \
		--ways $(WAYS) \
		--shots $(SHOTS) \
		--queries $(QUERIES) \
		--episodes $(EPISODES) \
		--eval-episodes $(EVAL_EPISODES) \
		--device $(DEVICE) $(EXTRA_ARGS) \
		--resume

# hyperparameter grid search for MobileNet variants (see cinic10-grid.py for options)
grid:
	PYTHONPATH=$(PYTHONPATH) $(UV) python -m cinic10.experiments.run_grid_search \
		--data-root $(DATA_ROOT) \
		--output-root $(OUTPUT_ROOT) \
		--seed $(SEED) \
		--device $(DEVICE) $(EXTRA_ARGS)

# resume/skip completed grid runs based on output directory structure
grid-resume:
	PYTHONPATH=$(PYTHONPATH) $(UV) python -m cinic10.experiments.run_grid_search \
		--data-root $(DATA_ROOT) \
		--output-root $(OUTPUT_ROOT) \
		--seed $(SEED) \
		--device $(DEVICE) $(EXTRA_ARGS) \
		--resume

# two-stage NAS with discrete architecture search followed by retraining
nas-two-stage:
	PYTHONPATH=$(PYTHONPATH) $(UV) python -m cinic10.experiments.run_nas_two_stage \
		--data-root $(DATA_ROOT) \
		--output-root $(OUTPUT_ROOT) \
		--seed $(SEED) \
		--epochs-search $(EPOCHS_SEARCH) \
		--epochs-retrain $(EPOCHS_RETRAIN) \
		--batch-size $(BATCH_SIZE) \
		--learning-rate $(LR) \
		--device $(DEVICE) $(EXTRA_ARGS)

# resume two-stage NAS from last checkpoint in output dir
nas-two-stage-resume:
	PYTHONPATH=$(PYTHONPATH) $(UV) python -m cinic10.experiments.run_nas_two_stage \
		--data-root $(DATA_ROOT) \
		--output-root $(OUTPUT_ROOT) \
		--seed $(SEED) \
		--epochs-search $(EPOCHS_SEARCH) \
		--epochs-retrain $(EPOCHS_RETRAIN) \
		--batch-size $(BATCH_SIZE) \
		--learning-rate $(LR) \
		--device $(DEVICE) $(EXTRA_ARGS) \
		--resume
