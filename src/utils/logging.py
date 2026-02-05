from __future__ import annotations

import logging


def get_logger(name: str = "eeg_genaug") -> logging.Logger:
    """Create or return a configured logger.

    Inputs:
    - name: logger name.

    Outputs:
    - logging.Logger instance with a stream handler.

    Internal logic:
    - Reuses existing handler if present; otherwise installs a simple formatter.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
