"""Few-shot learning modules."""

from cinic10.fewshot.protonet import PrototypicalNetwork, train_protonet

__all__: list[str] = ["PrototypicalNetwork", "train_protonet"]
