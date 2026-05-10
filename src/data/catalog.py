from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


SPLITS = ("train", "validation", "test")
SCENARIOS = ("attack", "normal")


@dataclass(frozen=True)
class DatasetRecord:
    family: str
    split: str
    scenario: str
    dataset_name: str
    path: Path

    def to_dict(self) -> dict[str, str]:
        item = asdict(self)
        item["path"] = str(self.path)
        return item


def build_dataset_catalog(config: dict) -> list[DatasetRecord]:
    data_root = Path(config["data_root"])
    records: list[DatasetRecord] = []
    for family in config["families"]:
        family_root = data_root / config["family_paths"][family]
        for scenario in SCENARIOS:
            for split in SPLITS:
                split_dir = family_root / scenario / split
                for csv_path in sorted(split_dir.glob("*.csv")):
                    records.append(
                        DatasetRecord(
                            family=family,
                            split=split,
                            scenario=scenario,
                            dataset_name=csv_path.stem,
                            path=csv_path,
                        )
                    )
    return records


def filter_catalog(
    catalog: Iterable[DatasetRecord],
    family: str | None = None,
    split: str | None = None,
    scenario: str | None = None,
) -> list[DatasetRecord]:
    records = list(catalog)
    if family is not None:
        records = [record for record in records if record.family == family]
    if split is not None:
        records = [record for record in records if record.split == split]
    if scenario is not None:
        records = [record for record in records if record.scenario == scenario]
    return records


def catalog_to_frame(catalog: Iterable[DatasetRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.to_dict() for record in catalog])


def summarize_catalog(catalog: Iterable[DatasetRecord]) -> pd.DataFrame:
    frame = catalog_to_frame(catalog)
    if frame.empty:
        return pd.DataFrame(columns=["family", "split", "scenario", "count"])
    return (
        frame.groupby(["family", "split", "scenario"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["family", "split", "scenario"])
    )
