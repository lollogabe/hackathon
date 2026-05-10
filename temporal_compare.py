from __future__ import annotations

import argparse
import logging

from src.evaluation.temporal_comparison import generate_temporal_comparison_plots
from src.utils.config import configure_logging, load_config
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot temporal score overlays for the quantum architecture selected in config.json "
            "and the best online baseline."
        )
    )
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument(
        "--architecture",
        choices=["zz_linear", "vqc", "quantum_kernel"],
        help="Optional override for config.json quantum_architecture.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep older temporal_compare*.png files instead of replacing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.architecture is not None:
        config["quantum_architecture"] = args.architecture
    config["reuse_quantum_cache"] = True
    configure_logging(config)
    set_global_seed(int(config["seed"]))
    logger = logging.getLogger(__name__)
    logger.info(
        "Plotting temporal comparisons for quantum_architecture=%s",
        config["quantum_architecture"],
    )
    written = generate_temporal_comparison_plots(
        config,
        clean_existing=not args.keep_existing,
    )
    for path in written:
        logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
