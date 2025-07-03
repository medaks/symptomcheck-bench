import logging
import logging.config
from typing import Optional

_LOGGING_CONFIG = {
    "formatters": {
        "standard": {"format": "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"}
    },
    "loggers": {"": {"propagate": True, "level": "INFO", "handlers": ["console"]}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "standard"}},
    "version": 1,
    "disable_existing_loggers": False,
    "root": {"level": "INFO", "handlers": ["console"]},
}


def get_logger(name: Optional[str] = None) -> "logging.Logger":
    logging.config.dictConfig(_LOGGING_CONFIG)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return logging.getLogger(name=name)
