"""Run one supervised CINIC-10 experiment."""

import argparse
import logging
from pathlib import Path
from typing import get_args

import torch

from cinic10.config import ArchitectureName, TrainingConfig
from cinic10.data import create_dataloader, resolve_data_root
from cinic10.models import create_model
from cinic10.training.engine import evaluate, fit
from cinic10.training.optimizer import create_optimizers, create_scheduler
from cinic10.utils import dump_json, pick_device, set_seed

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train CNN model on CINIC-10")
    architecture_choices = tuple(get_args(ArchitectureName))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--architecture",
        type=str,
        default="mobilenet_v3_small",
        choices=architecture_choices,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--arch-learning-rate", type=float, default=3e-4)
    parser.add_argument("--arch-weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--no-autoaugment", action="store_true")
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--cutmix", action="store_true")
    parser.add_argument("--mix-alpha", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--nas-entropy-weight", type=float, default=1e-3)
    parser.add_argument("--nas-temperature-start", type=float, default=5.0)
    parser.add_argument("--nas-temperature-end", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Main CLI function."""
    args = _parse_args()

    config = TrainingConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        architecture=args.architecture,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        arch_learning_rate=args.arch_learning_rate,
        arch_weight_decay=args.arch_weight_decay,
        dropout=args.dropout,
        num_workers=args.num_workers,
        use_autoaugment=not args.no_autoaugment,
        use_mixup=args.mixup,
        use_cutmix=args.cutmix,
        mix_alpha=args.mix_alpha,
        pretrained=args.pretrained,
        train_fraction=args.train_fraction,
        checkpoint_interval=args.checkpoint_interval,
        nas_entropy_weight=args.nas_entropy_weight,
        nas_temperature_start=args.nas_temperature_start,
        nas_temperature_end=args.nas_temperature_end,
        device=args.device,
    )

    logger.info(
        "Starting training run: architecture=%s seed=%d output=%s",
        config.architecture,
        config.seed,
        config.output_dir,
    )
    set_seed(config.seed)
    device = pick_device(config.device)
    logger.info("Resolved device: %s", device)
    data_root = resolve_data_root(config.data_root)
    logger.debug("Resolved data root: %s", data_root)

    train_loader = create_dataloader(
        data_root=data_root,
        split="train",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_autoaugment=config.use_autoaugment,
        train_fraction=config.train_fraction,
        seed=config.seed,
    )
    val_loader = create_dataloader(
        data_root=data_root,
        split="validate",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_autoaugment=False,
        train_fraction=1.0,
        seed=config.seed,
    )
    test_loader = create_dataloader(
        data_root=data_root,
        split="test",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        use_autoaugment=False,
        train_fraction=1.0,
        seed=config.seed,
    )

    model = create_model(
        architecture=config.architecture,
        num_classes=10,
        dropout=config.dropout,
        pretrained=config.pretrained,
    ).to(device)
    logger.info("Model created: %s", config.architecture)

    optimizers = create_optimizers(model, config)
    scheduler = create_scheduler(optimizers.weight_optimizer, config.epochs)

    logger.info("Beginning fit")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizers.weight_optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        resume=args.resume,
        verbose=not args.quiet,
        architecture_optimizer=optimizers.architecture_optimizer,
        architecture_loader=val_loader,
    )

    logger.info("Fit finished")

    best_checkpoint = config.output_dir / "best.pt"
    if best_checkpoint.exists():
        state = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        progress_desc="test",
        verbose=not args.quiet,
    )
    logger.info("Test metrics: loss=%s accuracy=%s", test_metrics.loss, test_metrics.accuracy)
    dump_json(
        config.output_dir / "test_metrics.json",
        {"test_loss": test_metrics.loss, "test_accuracy": test_metrics.accuracy},
    )


if __name__ == "__main__":
    main()
