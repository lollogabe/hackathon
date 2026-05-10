from __future__ import annotations

import argparse
import logging

from src.evaluation.evaluator import evaluate_family
from src.utils.config import configure_logging, ensure_output_dirs, load_config, resolve_device
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved quantum DDoS detectors.")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config)
    ensure_output_dirs(config)
    set_global_seed(int(config["seed"]))
    logging.getLogger(__name__).info("Using device: %s", resolve_device(config))
    for family in config["families"]:
        evaluate_family(config, family)


if __name__ == "__main__":
    main()
