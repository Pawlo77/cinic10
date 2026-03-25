"""Train ConvKAN-converted NAS architecture from scratch.

This runner loads discrete NAS operation choices from a completed NAS search
artifact and trains a new model where eligible Conv2d layers are replaced with
ConvKAN layers.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import get_args

import torch

from cinic10.config import AugmentationMode, OptimizerName, TrainingConfig
from cinic10.data import create_dataloader, resolve_data_root
from cinic10.models.factory import replace_conv2d_with_convkan
from cinic10.models.nas_cnn import DiscreteNasCnn
from cinic10.training.engine import evaluate, fit
from cinic10.training.optimizer import create_optimizers, create_scheduler
from cinic10.utils import dump_json, pick_device, set_seed

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train ConvKAN NAS model from searched architecture"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--nas-output-root",
        type=Path,
        default=Path("outputs/03_models/nas_two_stage"),
        help="Root directory with NAS outputs containing seed_*/search/architecture.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    optimizer_choices = tuple(get_args(OptimizerName))
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=optimizer_choices,
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    augmentation_choices = tuple(get_args(AugmentationMode))
    parser.add_argument(
        "--augmentation",
        type=str,
        default="standard",
        choices=augmentation_choices,
    )
    parser.add_argument("--mix-alpha", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument(
        "--convkan-min-kernel-size",
        type=int,
        default=5,
        help="Replace only Conv2d layers with kernel size >= this value",
    )
    parser.add_argument(
        "--convkan-max-channels",
        type=int,
        default=64,
        help="Replace only Conv2d layers with in/out channels <= this value",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping based on validation loss",
    )
    return parser.parse_args()


def _load_selected_operations(architecture_path: Path) -> list[str]:
    """Load selected operation names from NAS search artifact."""
    payload = json.loads(architecture_path.read_text(encoding="utf-8"))
    selected_names = payload.get("selected_operation_names", [])
    if selected_names:
        return [str(name) for name in selected_names]

    selected_ops = payload.get("selected_operations", [])
    parsed: list[str] = []
    for entry in selected_ops:
        text = str(entry)
        if ":" in text:
            op_name = text.split(":", maxsplit=1)[1].strip().split(" ", maxsplit=1)[0]
            parsed.append(op_name)

    if parsed:
        return parsed

    raise ValueError(f"Unable to read selected operations from {architecture_path}")


def main() -> None:
    """Load NAS architecture, convert to ConvKAN and train from scratch."""
    args = _parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    data_root = resolve_data_root(args.data_root)

    architecture_path = args.nas_output_root / f"seed_{args.seed}" / "search" / "architecture.json"
    if not architecture_path.exists():
        raise FileNotFoundError(
            "NAS architecture file not found. Expected: "
            f"{architecture_path}. Run nas-two-stage search first for this seed."
        )

    selected_operations = _load_selected_operations(architecture_path)

    config = TrainingConfig(
        data_root=data_root,
        output_dir=args.output_dir,
        architecture="nas_cnn",
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        num_workers=args.num_workers,
        augmentation=args.augmentation,
        mix_alpha=args.mix_alpha,
        train_fraction=args.train_fraction,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
    )

    logger.info(
        "run_nas_convkan: data=%s output=%s nas=%s seed=%d",
        data_root,
        args.output_dir,
        args.nas_output_root,
        args.seed,
    )

    train_loader = create_dataloader(
        data_root=data_root,
        split="train",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        augmentation=config.augmentation,
        train_fraction=config.train_fraction,
        seed=config.seed,
        mix_alpha=config.mix_alpha,
        num_classes=10,
    )
    val_loader = create_dataloader(
        data_root=data_root,
        split="validate",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        augmentation="none",
        seed=config.seed,
        mix_alpha=config.mix_alpha,
        num_classes=10,
    )
    test_loader = create_dataloader(
        data_root=data_root,
        split="test",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        augmentation="none",
        seed=config.seed,
        mix_alpha=config.mix_alpha,
        num_classes=10,
    )

    model = DiscreteNasCnn(
        selected_operations=selected_operations,
        num_classes=10,
        dropout=config.dropout,
    )
    model = replace_conv2d_with_convkan(
        model,
        min_kernel_size=args.convkan_min_kernel_size,
    ).to(device)

    optimizers = create_optimizers(model, config)
    scheduler = create_scheduler(optimizers.weight_optimizer, config.epochs)

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
        early_stopping=args.early_stopping,
    )

    best_checkpoint = config.output_dir / "best.pt"
    if best_checkpoint.exists():
        state = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        progress_desc="test",
        verbose=not args.quiet,
    )

    dump_json(
        config.output_dir / "test_metrics.json",
        {"test_loss": test_metrics.loss, "test_accuracy": test_metrics.accuracy},
    )
    dump_json(
        config.output_dir / "nas_convkan_source.json",
        {
            "seed": args.seed,
            "nas_output_root": str(args.nas_output_root),
            "architecture_path": str(architecture_path),
            "selected_operation_names": selected_operations,
            "convkan_min_kernel_size": args.convkan_min_kernel_size,
            "convkan_max_channels": args.convkan_max_channels,
        },
    )


if __name__ == "__main__":
    main()
