from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(root: Path) -> None:
    path = root / "logs" / "factory.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("nc_factory")
    logger.setLevel(logging.INFO)
    for existing in list(logger.handlers):
        if isinstance(existing, RotatingFileHandler):
            logger.removeHandler(existing)
            existing.close()
    handler = RotatingFileHandler(path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
