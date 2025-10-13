import logging
from datetime import UTC, datetime
from pathlib import Path

import structlog


class CustomLogger:
    def __init__(self, log_dir="logs"):
        self.logs_dir = Path.cwd() / log_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # timestamp log file (for persistence)
        log_file = f"{datetime.now(UTC).strftime('%Y_%m_%d_%H_%M_%S')}.log"
        self.log_file_path = self.logs_dir / log_file

    def get_logger(self, name=__file__):
        logger_name = Path(name).stem

        # configure logging for console + file (both JSON)
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))  # RAW JSON lines

        # console config
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[file_handler, console_handler],
        )
        # configure structlog for JSON structured logging
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger(logger_name)
