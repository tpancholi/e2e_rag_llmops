from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Hello from e2e-rag-llmops!")


if __name__ == "__main__":
    main()
