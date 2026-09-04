#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PU_ONLY_CRITERIA = (
    "crit_leeliu",
    "crit_puauc",
    "crit_nnpurisk",
    "crit_f1hat",
)


def lee_liu_from_scores(
    positive_scores: Sequence[float],
    unlabeled_scores: Sequence[float],
    threshold: float = 0.5,
) -> float:
    if not positive_scores or not unlabeled_scores:
        raise ValueError("both labeled-positive and unlabeled scores are required")
    recall_proxy = sum(float(value) > threshold for value in positive_scores) / len(
        positive_scores
    )
    predicted_positive_rate = sum(
        float(value) > threshold for value in unlabeled_scores
    ) / len(unlabeled_scores)
    return recall_proxy * recall_proxy / max(predicted_positive_rate, 1e-6)


def _tie_key(row: Mapping[str, str]) -> tuple[str, ...]:
    fields = ("lr", "wd", "hidden", "epoch", "cfg", "method")
    return tuple(str(row.get(field, "")) for field in fields)


def select_configurations(
    rows: Iterable[Mapping[str, str]],
    group_fields: Sequence[str],
    criterion: str = "crit_leeliu",
) -> list[dict[str, str]]:
    if criterion not in PU_ONLY_CRITERIA:
        raise ValueError(
            f"{criterion!r} is not P/U-only; choose one of {PU_ONLY_CRITERIA}"
        )
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        grouped[tuple(str(row.get(field, "")) for field in group_fields)].append(row)

    selected: list[dict[str, str]] = []
    for key in sorted(grouped):
        candidates: list[tuple[float, tuple[str, ...], dict[str, str]]] = []
        for row in grouped[key]:
            try:
                value = float(row.get(criterion, ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                candidates.append((value, _tie_key(row), row))
        if not candidates:
            raise ValueError(f"group {key!r} has no finite {criterion} value")


        best_value = max(item[0] for item in candidates)
        tied = [item for item in candidates if item[0] == best_value]
        selected.append(min(tied, key=lambda item: item[1])[2])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--group-by",
        nargs="+",
        required=True,
        help="fields defining one dataset/condition/seed/method cell",
    )
    parser.add_argument("--criterion", choices=PU_ONLY_CRITERIA, default="crit_leeliu")
    args = parser.parse_args()

    with args.input_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = reader.fieldnames or []
    chosen = select_configurations(rows, args.group_by, args.criterion)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(chosen)


if __name__ == "__main__":
    main()
