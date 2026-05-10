from __future__ import annotations

import argparse
import copy
import logging
import shutil
from pathlib import Path

from benchmarks import run_family_benchmarks
from src.evaluation.evaluator import evaluate_family
from src.evaluation.reporting import generate_reports
from src.evaluation.temporal_comparison import generate_temporal_comparison_plots
from src.training.trainer import train_family
from src.utils.config import configure_logging, ensure_output_dirs, load_config, resolve_device
from src.utils.seed import set_global_seed

ARCHITECTURES = ("zz_linear", "vqc", "quantum_kernel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate/report every quantum architecture into separate "
            "presentation output directories."
        )
    )
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=ARCHITECTURES,
        default=list(ARCHITECTURES),
        help="Architectures to run. Defaults to all three.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse existing model artifacts in each architecture directory.",
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Reuse existing results_{family}.json quantum metrics.",
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Reuse existing benchmark results in each architecture directory.",
    )
    parser.add_argument(
        "--skip-temporal",
        action="store_true",
        help="Skip temporal_compare presentation plots.",
    )
    parser.add_argument(
        "--keep-tables",
        action="store_true",
        help="Keep CSV/Markdown tables inside each architecture directory.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete each selected architecture output directory before running it.",
    )
    return parser.parse_args()


def architecture_config(base_config: dict, architecture: str) -> dict:
    config = copy.deepcopy(base_config)
    base_output = Path(base_config["output_root"])
    config["output_root"] = str(base_output / "architectures" / architecture)
    config["quantum_architecture"] = architecture
    config["report_detail_level"] = "presentation"
    config["include_individual_confusion_plots"] = False
    config["write_report_tables"] = False
    return config


def run_architecture(config: dict, args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    architecture = config["quantum_architecture"]
    output_root = Path(config["output_root"])
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    ensure_output_dirs(config)
    if args.keep_tables:
        config["write_report_tables"] = True
    elif (output_root / "tables").exists():
        shutil.rmtree(output_root / "tables")

    set_global_seed(int(config["seed"]))
    logger.info("Running architecture=%s output_root=%s", architecture, output_root)
    logger.info("Using device: %s", resolve_device(config))

    if not args.skip_train:
        for family in config["families"]:
            train_family(config, family)

    if not args.skip_evaluate:
        for family in config["families"]:
            evaluate_family(config, family)

    if not args.skip_benchmarks:
        for family in config["families"]:
            run_family_benchmarks(config, family)

    written = generate_reports(config)
    for path in written["plots"]:
        logger.info("wrote %s", path)

    if not args.skip_temporal:
        temporal_written = generate_temporal_comparison_plots(
            config,
            clean_existing=True,
            write_summary_table=bool(config["write_report_tables"]),
        )
        for path in temporal_written:
            logger.info("wrote %s", path)

    if not config["write_report_tables"] and (output_root / "tables").exists():
        shutil.rmtree(output_root / "tables")


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    configure_logging(base_config)
    logger = logging.getLogger(__name__)
    for architecture in args.architectures:
        config = architecture_config(base_config, architecture)
        run_architecture(config, args)
    logger.info("Completed architecture run set: %s", ", ".join(args.architectures))


if __name__ == "__main__":
    main()
