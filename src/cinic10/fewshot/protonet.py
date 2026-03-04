"""Prototypical network implementation for CINIC-10 few-shot experiments."""

import logging
from dataclasses import dataclass
from os import PathLike
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torchvision import datasets, transforms
from tqdm.auto import trange

from cinic10.config import FewShotConfig
from cinic10.data import resolve_data_root
from cinic10.utils import (
    atomic_torch_save,
    cpu_time_seconds,
    device_memory_snapshot,
    dump_json,
    ensure_dir,
    process_memory_snapshot,
    reset_device_peak_memory_stats,
    save_model_weights_optimized,
    synchronize_device,
    wall_time_seconds,
)

logger = logging.getLogger(__name__)


class ConvEmbedding(nn.Module):
    """Simple convolutional embedding network for few-shot learning.

    Args:
        embedding_dim: Output embedding dimension.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.project = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into embedding vectors.

        Args:
            x: Input images.

        Returns:
            Embedding tensor.
        """
        feat = self.backbone(x).flatten(1)
        return self.project(feat)


class PrototypicalNetwork(nn.Module):
    """Prototypical network classifier.

    Args:
        embedding_dim: Feature dimension.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = ConvEmbedding(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor into embeddings.

        Args:
            x: Input images.

        Returns:
            Embeddings.
        """
        return self.encoder(x)


def _transform() -> transforms.Compose:
    """Create a deterministic transform for episodic few-shot."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.47889522, 0.47227842, 0.43047404),
                std=(0.24205776, 0.23828046, 0.25874835),
            ),
        ]
    )


@dataclass(frozen=True, slots=True)
class Episode:
    """One few-shot learning episode."""

    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor


def _class_to_indices(targets: list[int], classes: int) -> dict[int, np.ndarray]:
    """Build index map per class.

    Args:
        targets: Integer labels.
        classes: Number of classes.

    Returns:
        Mapping from class id to index array.
    """
    labels = np.asarray(targets)
    return {class_id: np.where(labels == class_id)[0] for class_id in range(classes)}


def sample_episode(
    dataset: datasets.ImageFolder,
    ways: int,
    shots: int,
    queries: int,
    rng: np.random.Generator,
) -> Episode:
    """Sample one N-way K-shot episode.

    Args:
        dataset: Image folder dataset.
        ways: Number of classes.
        shots: Support images per class.
        queries: Query images per class.
        rng: NumPy random generator.

    Returns:
        Episodic support and query tensors.
    """
    class_map = _class_to_indices(dataset.targets, len(dataset.classes))
    selected_classes = rng.choice(len(dataset.classes), size=ways, replace=False)

    support_images: list[torch.Tensor] = []
    support_labels: list[int] = []
    query_images: list[torch.Tensor] = []
    query_labels: list[int] = []

    for episodic_class_id, class_id in enumerate(selected_classes):
        pool = class_map[int(class_id)]
        required = shots + queries
        chosen = rng.choice(pool, size=required, replace=False)
        support_idx = chosen[:shots]
        query_idx = chosen[shots:]

        for idx in support_idx:
            image, _ = dataset[int(idx)]
            support_images.append(image)
            support_labels.append(episodic_class_id)
        for idx in query_idx:
            image, _ = dataset[int(idx)]
            query_images.append(image)
            query_labels.append(episodic_class_id)

    return Episode(
        support_x=torch.stack(support_images),
        support_y=torch.tensor(support_labels, dtype=torch.long),
        query_x=torch.stack(query_images),
        query_y=torch.tensor(query_labels, dtype=torch.long),
    )


def _prototypical_logits(
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    ways: int,
) -> torch.Tensor:
    """Compute logits as negative distances to class prototypes.

    Args:
        support_embeddings: Support embeddings.
        support_labels: Support episodic labels.
        query_embeddings: Query embeddings.
        ways: Number of classes in episode.

    Returns:
        Query logits.
    """
    prototypes: list[torch.Tensor] = []
    for class_id in range(ways):
        cls_embed = support_embeddings[support_labels == class_id]
        prototypes.append(cls_embed.mean(dim=0))
    proto_tensor = torch.stack(prototypes)

    distances = torch.cdist(query_embeddings, proto_tensor, p=2)
    return -distances


def run_episode(
    model: PrototypicalNetwork,
    episode: Episode,
    ways: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Forward one episode and compute loss and accuracy.

    Args:
        model: Prototypical network.
        episode: Episode data.
        ways: Number of classes.
        device: Compute device.

    Returns:
        Loss tensor and episode accuracy.
    """
    support_x = episode.support_x.to(device)
    support_y = episode.support_y.to(device)
    query_x = episode.query_x.to(device)
    query_y = episode.query_y.to(device)

    support_embeddings = model(support_x)
    query_embeddings = model(query_x)

    logits = _prototypical_logits(support_embeddings, support_y, query_embeddings, ways)
    loss = nn.CrossEntropyLoss()(logits, query_y)
    acc = float((logits.argmax(dim=1) == query_y).float().mean().item())
    return loss, acc


def _evaluate(
    model: PrototypicalNetwork,
    dataset: datasets.ImageFolder,
    config: FewShotConfig,
    device: torch.device,
    rng: np.random.Generator,
) -> float:
    """Evaluate average episodic accuracy.

    Args:
        model: Prototypical network.
        dataset: Evaluation dataset.
        config: Few-shot config.
        device: Compute device.
        rng: Random generator.

    Returns:
        Mean accuracy.
    """
    model.eval()
    accs: list[float] = []
    with torch.no_grad():
        for _ in range(config.eval_episodes):
            episode = sample_episode(dataset, config.ways, config.shots, config.queries, rng)
            _, acc = run_episode(model, episode, config.ways, device)
            accs.append(acc)
    return float(np.mean(accs))


@dataclass(frozen=True, slots=True)
class FewShotCheckpointStore:
    """Persist and restore few-shot training state.

    Attributes:
        last_path: Path to the resumable checkpoint.
        best_path: Path to best-validation checkpoint.
    """

    last_path: PathLike[str] | str
    best_path: PathLike[str] | str

    def save(
        self,
        *,
        path: PathLike[str] | str,
        model: PrototypicalNetwork,
        optimizer: Adam,
        episode: int,
        best_val_accuracy: float,
        rng_state: dict[str, Any],
        status: str,
    ) -> None:
        """Write checkpoint state atomically.

        Args:
            path: Destination checkpoint file.
            model: Model to serialize.
            optimizer: Optimizer to serialize.
            episode: Last completed episode.
            best_val_accuracy: Best validation accuracy so far.
            rng_state: NumPy RNG state.
            status: Checkpoint status tag.
        """
        atomic_torch_save(
            {
                "episode": episode,
                "status": status,
                "best_val_accuracy": best_val_accuracy,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": rng_state,
            },
            path,
        )

    def load(
        self,
        *,
        path: PathLike[str] | str,
        model: PrototypicalNetwork,
        optimizer: Adam,
    ) -> tuple[int, float, dict[str, Any]]:
        """Load checkpoint state.

        Args:
            path: Source checkpoint file.
            model: Model target.
            optimizer: Optimizer target.

        Returns:
            Tuple of episode, best validation accuracy, and RNG state.
        """
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        return (
            int(state["episode"]),
            float(state.get("best_val_accuracy", 0.0)),
            dict(state["rng_state"]),
        )


def train_protonet(
    config: FewShotConfig,
    device: torch.device,
    resume: bool = False,
    verbose: bool = True,
) -> dict[str, float]:
    """Train and evaluate Prototypical Network on CINIC-10.

    Args:
        config: Few-shot config.
        device: Compute device.
        resume: Resume from checkpoint if present.

    Returns:
        Dictionary with best validation accuracy.
    """
    logger.info("train_protonet: output=%s device=%s", config.output_dir, device)
    ensure_dir(config.output_dir)
    data_root = resolve_data_root(config.data_root)

    train_set = datasets.ImageFolder(root=str(data_root / "train"), transform=_transform())
    val_set = datasets.ImageFolder(root=str(data_root / "validate"), transform=_transform())

    model = PrototypicalNetwork(embedding_dim=config.embedding_dim).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    last_checkpoint_path = config.output_dir / "fewshot_last.ckpt"
    best_checkpoint_path = config.output_dir / "fewshot_best.pt"
    checkpoint_store = FewShotCheckpointStore(
        last_path=last_checkpoint_path,
        best_path=best_checkpoint_path,
    )

    best_val = 0.0
    start_episode = 1
    episode_resource_stats: list[dict[str, Any]] = []

    if resume and last_checkpoint_path.exists():
        loaded_episode, best_val, restored_rng_state = checkpoint_store.load(
            path=checkpoint_store.last_path,
            model=model,
            optimizer=optimizer,
        )
        rng.bit_generator.state = restored_rng_state
        start_episode = loaded_episode + 1

    if start_episode <= config.episodes:
        last_completed_episode = start_episode - 1
        try:
            episode_progress = trange(
                start_episode,
                config.episodes + 1,
                desc="few-shot-episodes",
                leave=True,
                disable=not verbose,
            )
            for episode_idx in episode_progress:
                synchronize_device(device)
                reset_device_peak_memory_stats(device)
                episode_wall_start = wall_time_seconds()
                episode_cpu_start = cpu_time_seconds()
                memory_start = process_memory_snapshot()

                model.train()
                episode = sample_episode(train_set, config.ways, config.shots, config.queries, rng)
                loss, _ = run_episode(model, episode, config.ways, device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                val_acc: float | None = None

                if (
                    episode_idx % max(1, config.eval_interval) == 0
                    or episode_idx == config.episodes
                ):
                    val_acc = _evaluate(model, val_set, config, device, rng)
                    if val_acc > best_val:
                        best_val = val_acc
                        checkpoint_store.save(
                            path=checkpoint_store.best_path,
                            model=model,
                            optimizer=optimizer,
                            episode=episode_idx,
                            best_val_accuracy=best_val,
                            rng_state=dict(rng.bit_generator.state),
                            status="best",
                        )
                        save_model_weights_optimized(
                            model, config.output_dir / "fewshot_best.safetensors"
                        )

                synchronize_device(device)
                episode_wall_end = wall_time_seconds()
                episode_cpu_end = cpu_time_seconds()
                memory_end = process_memory_snapshot()
                accelerator_memory = device_memory_snapshot(device)
                episode_resource_stats.append(
                    {
                        "episode": episode_idx,
                        "wall_time_seconds": float(episode_wall_end - episode_wall_start),
                        "cpu_time_seconds": float(episode_cpu_end - episode_cpu_start),
                        "ram_current_bytes_start": memory_start["ram_current_bytes"],
                        "ram_current_bytes_end": memory_end["ram_current_bytes"],
                        "ram_peak_bytes_end": memory_end["ram_peak_bytes"],
                        "device_type": device.type,
                        "train_loss": float(loss.item()),
                        "val_accuracy": val_acc,
                        **accelerator_memory,
                    }
                )
                if verbose:
                    episode_progress.set_postfix(
                        {
                            "train_loss": f"{loss.item():.4f}",
                            "best_val_acc": f"{best_val:.4f}",
                            "wall_s": f"{episode_resource_stats[-1]['wall_time_seconds']:.2f}",
                        }
                    )

                if episode_idx % max(1, config.checkpoint_interval) == 0:
                    checkpoint_store.save(
                        path=checkpoint_store.last_path,
                        model=model,
                        optimizer=optimizer,
                        episode=episode_idx,
                        best_val_accuracy=best_val,
                        rng_state=dict(rng.bit_generator.state),
                        status="running",
                    )
                    dump_json(
                        config.output_dir / "fewshot_episode_resource_stats.json",
                        episode_resource_stats,
                    )
                last_completed_episode = episode_idx
        except KeyboardInterrupt:
            checkpoint_store.save(
                path=checkpoint_store.last_path,
                model=model,
                optimizer=optimizer,
                episode=max(last_completed_episode, 0),
                best_val_accuracy=best_val,
                rng_state=dict(rng.bit_generator.state),
                status="interrupted",
            )
            logger.warning("few-shot training interrupted at episode %d", last_completed_episode)
            raise

    metrics = {"best_val_accuracy": best_val}
    dump_json(config.output_dir / "fewshot_metrics.json", metrics)
    if episode_resource_stats:
        dump_json(
            config.output_dir / "fewshot_episode_resource_stats.json",
            episode_resource_stats,
        )
    checkpoint_store.save(
        path=checkpoint_store.last_path,
        model=model,
        optimizer=optimizer,
        episode=config.episodes,
        best_val_accuracy=best_val,
        rng_state=dict(rng.bit_generator.state),
        status="completed",
    )
    atomic_torch_save(model.state_dict(), config.output_dir / "protonet.pt")
    return metrics
