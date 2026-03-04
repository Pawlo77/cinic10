"""Run two-stage NAS: search then retrain discrete architecture."""

import argparse
import json
import logging
from pathlib import Path

import torch

from cinic10.config import TrainingConfig
from cinic10.data import create_dataloader, resolve_data_root
from cinic10.models import create_model
from cinic10.models.nas_cnn import DiscreteNasCnn
from cinic10.training.engine import evaluate, fit
from cinic10.training.optimizer import create_optimizers, create_scheduler
from cinic10.utils import dump_json, pick_device, set_seed

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for two-stage NAS workflow."""
    parser = argparse.ArgumentParser(description="Run two-stage NAS on CINIC-10")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs-search", type=int, default=30)
    parser.add_argument("--epochs-retrain", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--arch-learning-rate", type=float, default=3e-4)
    parser.add_argument("--arch-weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pretrained", action="store_true")
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


def _load_selected_operations(architecture_path: Path) -> list[str]:
    """Load selected operation names from NAS search artifact.

    Args:
        architecture_path: Path to `architecture.json` produced by search stage.

    Returns:
        Ordered operation names for all NAS edges.
    """
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
    """Run NAS search and retrain discrete architecture."""
    args = _parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    data_root = resolve_data_root(args.data_root)
    logger.info(
        "run_nas_two_stage: data=%s output=%s device=%s", data_root, args.output_root, device
    )

    search_output = args.output_root / "search"
    retrain_output = args.output_root / "retrain"

    train_loader = create_dataloader(
        data_root=data_root,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_autoaugment=True,
        train_fraction=1.0,
        seed=args.seed,
    )
    val_loader = create_dataloader(
        data_root=data_root,
        split="validate",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_autoaugment=False,
        train_fraction=1.0,
        seed=args.seed,
    )
    test_loader = create_dataloader(
        data_root=data_root,
        split="test",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_autoaugment=False,
        train_fraction=1.0,
        seed=args.seed,
    )

    search_config = TrainingConfig(
        data_root=data_root,
        output_dir=search_output,
        architecture="nas_cnn",
        seed=args.seed,
        epochs=args.epochs_search,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        arch_learning_rate=args.arch_learning_rate,
        arch_weight_decay=args.arch_weight_decay,
        dropout=args.dropout,
        num_workers=args.num_workers,
        use_mixup=args.mixup,
        use_cutmix=args.cutmix,
        mix_alpha=args.mix_alpha,
        pretrained=args.pretrained,
        checkpoint_interval=args.checkpoint_interval,
        nas_entropy_weight=args.nas_entropy_weight,
        nas_temperature_start=args.nas_temperature_start,
        nas_temperature_end=args.nas_temperature_end,
        device=args.device,
    )

    search_model = create_model(
        architecture=search_config.architecture,
        num_classes=10,
        dropout=search_config.dropout,
        pretrained=False,
    ).to(device)
    search_optimizers = create_optimizers(search_model, search_config)
    search_scheduler = create_scheduler(search_optimizers.weight_optimizer, search_config.epochs)

    search_metrics = fit(
        model=search_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=search_optimizers.weight_optimizer,
        scheduler=search_scheduler,
        config=search_config,
        device=device,
        resume=args.resume,
        verbose=not args.quiet,
        architecture_optimizer=search_optimizers.architecture_optimizer,
        architecture_loader=val_loader,
    )

    architecture_path = search_output / "architecture.json"
    selected_operations = _load_selected_operations(architecture_path)

    retrain_config = TrainingConfig(
        data_root=data_root,
        output_dir=retrain_output,
        architecture="nas_cnn",
        seed=args.seed,
        epochs=args.epochs_retrain,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        num_workers=args.num_workers,
        use_mixup=args.mixup,
        use_cutmix=args.cutmix,
        mix_alpha=args.mix_alpha,
        pretrained=args.pretrained,
        checkpoint_interval=args.checkpoint_interval,
        nas_entropy_weight=0.0,
        nas_temperature_start=1.0,
        nas_temperature_end=1.0,
        device=args.device,
    )

    retrain_model = DiscreteNasCnn(
        selected_operations=selected_operations,
        num_classes=10,
        dropout=args.dropout,
    ).to(device)
    retrain_optimizers = create_optimizers(retrain_model, retrain_config)
    retrain_scheduler = create_scheduler(retrain_optimizers.weight_optimizer, retrain_config.epochs)

    retrain_metrics = fit(
        model=retrain_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=retrain_optimizers.weight_optimizer,
        scheduler=retrain_scheduler,
        config=retrain_config,
        device=device,
        resume=args.resume,
        verbose=not args.quiet,
    )

    best_checkpoint = retrain_output / "best.pt"
    if best_checkpoint.exists():
        state = torch.load(best_checkpoint, map_location=device)
        retrain_model.load_state_dict(state["model_state_dict"])

    test_metrics = evaluate(
        model=retrain_model,
        dataloader=test_loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device=device,
        progress_desc="test",
        verbose=not args.quiet,
    )

    summary = {
        "search": search_metrics,
        "retrain": retrain_metrics,
        "selected_operation_names": selected_operations,
        "test_loss": test_metrics.loss,
        "test_accuracy": test_metrics.accuracy,
    }
    dump_json(args.output_root / "two_stage_summary.json", summary)


if __name__ == "__main__":
    main()
