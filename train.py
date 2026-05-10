from __future__ import annotations

import argparse
import logging

from src.data.catalog import build_dataset_catalog, summarize_catalog
from src.training.trainer import train_family
from src.utils.config import configure_logging, ensure_output_dirs, load_config, resolve_device
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the quantum-inspired DDoS detector.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset catalog without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config)
    ensure_output_dirs(config)
    set_global_seed(int(config["seed"]))
    logger = logging.getLogger(__name__)
    logger.info("Using device: %s", resolve_device(config))

    catalog = build_dataset_catalog(config)
    summary = summarize_catalog(catalog)
    print(summary.to_string(index=False))
    if args.dry_run:
        return
    for family in config["families"]:
        train_family(config, family)


if __name__ == "__main__":
    main()
