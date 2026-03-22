"""Run MobileNetV3-Small hyperparameter grid search."""

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from cinic10.config import build_mobilenet_grid
from cinic10.data import create_dataloader, resolve_data_root
from cinic10.models import create_model
from cinic10.training.engine import evaluate, fit
from cinic10.training.optimizer import create_optimizer, create_scheduler
from cinic10.utils import dump_json, pick_device, set_seed

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Run MobileNetV3-Small grid search")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--stop-after", type=int, default=None, help="Optional run_id to stop after (inclusive)"
    )
    return parser.parse_args()


def main() -> None:
    """Execute all grid search runs and save summary CSV."""
    args = _parse_args()
    resolved_data_root = resolve_data_root(args.data_root)
    logger.info("run_grid_search: data=%s output=%s", resolved_data_root, args.output_root)
    runs = build_mobilenet_grid(
        resolved_data_root,
        args.output_root,
        seed=args.seed,
        epochs=args.epochs,
    )
    device = pick_device(args.device)

    rows: list[dict[str, float | int | str]] = []

    if args.stop_after is not None:
        runs = runs[: args.stop_after + 1]

    run_progress = tqdm(
        enumerate(runs),
        total=len(runs),
        desc="grid-runs",
        leave=True,
        disable=args.quiet,
    )
    for run_id, config in run_progress:
        if not args.quiet:
            run_progress.set_postfix(
                {
                    "run": run_id,
                    "seed": config.seed,
                    "opt": config.optimizer,
                    "bs": config.batch_size,
                    "dropout": config.dropout,
                    "wd": config.weight_decay,
                }
            )
        logger.info("Starting run %d with config: %s", run_id, config)

        set_seed(config.seed)
        run_config = replace(
            config,
            num_workers=args.num_workers,
            device=args.device,
            checkpoint_interval=args.checkpoint_interval,
            pretrained=False,
            augmentation="none",
        )

        metrics_path = run_config.output_dir / "metrics.json"
        if args.resume and metrics_path.exists() and not args.force_rerun:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            test_metrics_path = run_config.output_dir / "test_metrics.json"
            if test_metrics_path.exists():
                test_metrics_payload = json.loads(test_metrics_path.read_text(encoding="utf-8"))
                test_loss = float(test_metrics_payload.get("test_loss", float("inf")))
                test_accuracy = float(test_metrics_payload.get("test_accuracy", 0.0))

            else:
                model = create_model(
                    architecture=run_config.architecture,
                    num_classes=10,
                    dropout=run_config.dropout,
                    pretrained=run_config.pretrained,
                ).to(device)
                test_loader = create_dataloader(
                    data_root=run_config.data_root,
                    split="test",
                    batch_size=run_config.batch_size,
                    num_workers=run_config.num_workers,
                    augmentation="autoaugment",
                    seed=run_config.seed,
                )

                best_checkpoint = run_config.output_dir / "best.pt"
                if best_checkpoint.exists():
                    state = torch.load(best_checkpoint, map_location=device, weights_only=False)
                    model.load_state_dict(state["model_state_dict"])
                evaluated = evaluate(
                    model=model,
                    dataloader=test_loader,
                    criterion=torch.nn.CrossEntropyLoss(),
                    device=device,
                    progress_desc="test",
                    verbose=not args.quiet,
                )
                test_loss = evaluated.loss
                test_accuracy = evaluated.accuracy

                dump_json(
                    test_metrics_path,
                    {"test_loss": test_loss, "test_accuracy": test_accuracy},
                )

                del model, test_loader
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.mps.is_available():
                    torch.mps.empty_cache()

            if not args.quiet:
                run_progress.set_postfix(
                    {
                        "run": run_id,
                        "status": "resumed",
                        "best_val_acc": f"{float(metrics.get('best_val_accuracy', 0.0)):.4f}",
                    }
                )
            rows.append(
                {
                    "run_id": run_id,
                    "seed": run_config.seed,
                    "optimizer": run_config.optimizer,
                    "batch_size": run_config.batch_size,
                    "dropout": run_config.dropout,
                    "weight_decay": run_config.weight_decay,
                    "best_val_accuracy": metrics.get("best_val_accuracy", 0.0),
                    "best_val_loss": metrics.get("best_val_loss", float("inf")),
                    "test_accuracy": test_accuracy,
                    "test_loss": test_loss,
                }
            )

            continue

        train_loader = create_dataloader(
            data_root=run_config.data_root,
            split="train",
            batch_size=run_config.batch_size,
            num_workers=run_config.num_workers,
            augmentation=run_config.augmentation,
            train_fraction=run_config.train_fraction,
            seed=run_config.seed,
            mix_alpha=run_config.mix_alpha,
            num_classes=10,
        )
        val_loader = create_dataloader(
            data_root=run_config.data_root,
            split="validate",
            batch_size=run_config.batch_size,
            num_workers=run_config.num_workers,
            augmentation="none",
            seed=run_config.seed,
            mix_alpha=run_config.mix_alpha,
            num_classes=10,
        )
        test_loader = create_dataloader(
            data_root=run_config.data_root,
            split="test",
            batch_size=run_config.batch_size,
            num_workers=run_config.num_workers,
            augmentation="none",
            seed=run_config.seed,
            mix_alpha=run_config.mix_alpha,
            num_classes=10,
        )

        model = create_model(
            architecture=run_config.architecture,
            num_classes=10,
            dropout=run_config.dropout,
            pretrained=run_config.pretrained,
        ).to(device)
        optimizer = create_optimizer(model, run_config)
        scheduler = create_scheduler(optimizer, run_config.epochs)

        metrics = fit(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            run_config,
            device,
            resume=args.resume,
            verbose=not args.quiet,
        )
        best_checkpoint = run_config.output_dir / "best.pt"
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
            run_config.output_dir / "test_metrics.json",
            {
                "test_loss": test_metrics.loss,
                "test_accuracy": test_metrics.accuracy,
            },
        )
        if not args.quiet:
            run_progress.set_postfix(
                {
                    "run": run_id,
                    "status": "done",
                    "best_val_acc": f"{float(metrics['best_val_accuracy']):.4f}",
                }
            )

        rows.append(
            {
                "run_id": run_id,
                "seed": run_config.seed,
                "optimizer": run_config.optimizer,
                "batch_size": run_config.batch_size,
                "dropout": run_config.dropout,
                "weight_decay": run_config.weight_decay,
                "best_val_accuracy": metrics["best_val_accuracy"],
                "best_val_loss": metrics["best_val_loss"],
                "test_accuracy": test_metrics.accuracy,
                "test_loss": test_metrics.loss,
            }
        )

        del model, optimizer, scheduler, train_loader, val_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.mps.is_available():
            torch.mps.empty_cache()

    logger.info("Grid search completed with %d runs. Saving summary CSV and JSON.", len(rows))
    summary = pd.DataFrame(rows)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_root / "grid_results.csv", index=False)
    dump_json(args.output_root / "grid_results.json", summary.to_dict(orient="records"))
    logger.info("All done! Summary saved to %s", args.output_root)


if __name__ == "__main__":
    main()
