from __future__ import annotations

import argparse
import logging

from src.evaluation.temporal_comparison import generate_temporal_comparison_plots
from src.utils.config import configure_logging, load_config
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay Quantum ZZ temporal scores with the best online model."
    )
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["reuse_quantum_cache"] = True
    configure_logging(config)
    set_global_seed(int(config["seed"]))
    written = generate_temporal_comparison_plots(config)
    logger = logging.getLogger(__name__)
    for path in written:
        logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
