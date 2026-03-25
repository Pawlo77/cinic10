"""CINIC-10 research framework.

This module configures package-wide logging on import. The log level can be
controlled with the environment variable `CINIC10_LOG_LEVEL` (preferred) or
`LOG_LEVEL` as a fallback. If the value is invalid or unset, `INFO` is used.
Logs are emitted to stdout and also written to `logs/cinic10.log` (or the
directory set by `CINIC10_LOG_DIR`).
"""

import logging
import multiprocessing
import os
from pathlib import Path

# Configure package-wide logging from environment
_env_level = os.environ.get("CINIC10_LOG_LEVEL") or os.environ.get("LOG_LEVEL")
_level_name = (_env_level or "INFO").upper()
_level = getattr(logging, _level_name, logging.INFO)

_repo_root = Path(__file__).resolve().parents[2]
_log_dir = Path(os.environ.get("CINIC10_LOG_DIR", str(_repo_root / "logs")))
_log_file_name = os.environ.get("CINIC10_LOG_FILE_NAME", "cinic10.log")
_log_file = _log_dir / _log_file_name

_handlers: list[logging.Handler] = [logging.StreamHandler()]
_file_handler_error: OSError | None = None
try:
    _log_dir.mkdir(parents=True, exist_ok=True)
    _handlers.append(logging.FileHandler(_log_file, encoding="utf-8"))
except OSError as exc:
    # Keep console logging even if file logging cannot be configured.
    _file_handler_error = exc

logging.basicConfig(
    level=_level,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=_handlers,
)

logger = logging.getLogger("cinic10")
if multiprocessing.current_process().name == "MainProcess":
    logger.info("Logging to file: %s", _log_file)
logger.debug("cinic10 root logger configured at level %s", logging.getLevelName(_level))
if _file_handler_error is None:
    logger.debug("cinic10 file logging enabled at %s", _log_file)
else:
    logger.warning("cinic10 file logging disabled: %s", _file_handler_error)

from cinic10.config import ArchitectureName, TrainingConfig  # noqa: E402

__all__: list[str] = ["ArchitectureName", "TrainingConfig"]
