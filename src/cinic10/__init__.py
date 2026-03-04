"""CINIC-10 research framework.

This module configures package-wide logging on import. The log level can be
controlled with the environment variable `CINIC10_LOG_LEVEL` (preferred) or
`LOG_LEVEL` as a fallback. If the value is invalid or unset, `INFO` is used.
"""

import logging
import os

# Configure package-wide logging from environment
_env_level = os.environ.get("CINIC10_LOG_LEVEL") or os.environ.get("LOG_LEVEL")
_level_name = (_env_level or "INFO").upper()
_level = getattr(logging, _level_name, logging.INFO)

logging.basicConfig(
    level=_level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logger = logging.getLogger("cinic10")
logger.debug("cinic10 root logger configured at level %s", logging.getLevelName(_level))

from cinic10.config import ArchitectureName, TrainingConfig  # noqa: E402

__all__: list[str] = ["ArchitectureName", "TrainingConfig"]
