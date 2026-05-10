from __future__ import annotations

import argparse
import logging

from src.evaluation.reporting import generate_reports
from src.utils.config import configure_logging, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate presentation-ready result tables and plots."
    )
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Generate presentation plots and remove/skip outputs/tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.plots_only:
        config["write_report_tables"] = False
    configure_logging(config)
    written = generate_reports(config)
    logger = logging.getLogger(__name__)
    for kind, paths in written.items():
        logger.info("%s:", kind)
        for path in paths:
            logger.info("  %s", path)


if __name__ == "__main__":
    main()
