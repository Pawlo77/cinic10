"""Run few-shot Prototypical Network experiment."""

import argparse
import logging
from pathlib import Path

from cinic10.config import FewShotConfig
from cinic10.fewshot.protonet import train_protonet
from cinic10.utils import pick_device, set_seed

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Train Prototypical Network on CINIC-10")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ways", type=int, default=5)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--queries", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=400)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Main CLI entrypoint."""
    args = _parse_args()
    config = FewShotConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        seed=args.seed,
        ways=args.ways,
        shots=args.shots,
        queries=args.queries,
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        device=args.device,
    )

    set_seed(config.seed)
    device = pick_device(config.device)
    logger.info("run_fewshot: starting few-shot run output=%s device=%s", config.output_dir, device)
    metrics = train_protonet(config, device, resume=args.resume, verbose=not args.quiet)
    print(metrics)


if __name__ == "__main__":
    main()
