"""MixUp and CutMix batch transformations."""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BatchMixer:
    """Apply MixUp or CutMix for classification batches.

    Args:
        alpha: Beta distribution parameter.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def _sample_lambda(self) -> float:
        """Sample interpolation coefficient from Beta(alpha, alpha)."""
        if self.alpha <= 0.0:
            return 1.0
        return float(np.random.beta(self.alpha, self.alpha))

    def mixup(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply MixUp augmentation.

        Args:
            images: Input image batch.
            labels: Input label tensor.

        Returns:
            Tuple of mixed images, primary labels, secondary labels and lambda.
        """
        lam = self._sample_lambda()
        logger.debug("mixup: lam=%.4f batch=%d", lam, images.size(0))
        index = torch.randperm(images.size(0), device=images.device)
        mixed = lam * images + (1.0 - lam) * images[index]
        return mixed, labels, labels[index], lam

    def cutmix(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation.

        Args:
            images: Input image batch.
            labels: Input label tensor.

        Returns:
            Tuple of mixed images, primary labels, secondary labels and lambda.
        """
        lam = self._sample_lambda()
        logger.debug("cutmix: lam=%.4f batch=%d", lam, images.size(0))
        batch_size, _, height, width = images.shape
        index = torch.randperm(batch_size, device=images.device)

        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(width * cut_ratio)
        cut_h = int(height * cut_ratio)

        cx = np.random.randint(width)
        cy = np.random.randint(height)

        x1 = int(np.clip(cx - cut_w // 2, 0, width))
        y1 = int(np.clip(cy - cut_h // 2, 0, height))
        x2 = int(np.clip(cx + cut_w // 2, 0, width))
        y2 = int(np.clip(cy + cut_h // 2, 0, height))

        patched = images.clone()
        patched[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        area = (x2 - x1) * (y2 - y1)
        adjusted_lam = 1.0 - area / float(height * width)
        return patched, labels, labels[index], adjusted_lam


def mixed_loss(
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
    criterion: torch.nn.Module,
) -> torch.Tensor:
    """Compute convex combination of losses for mixed labels.

    Args:
        logits: Logits tensor.
        labels_a: Primary labels.
        labels_b: Secondary labels.
        lam: Interpolation coefficient.
        criterion: Base loss function.

    Returns:
        Mixed loss value.
    """
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
