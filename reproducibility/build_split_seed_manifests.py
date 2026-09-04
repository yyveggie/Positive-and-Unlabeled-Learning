#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "experiments" / "src"
GENERATED = ROOT / "reproducibility" / "generated"
sys.path.insert(0, str(SRC))

from pu_lib import load_dataset, make_pu


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for source in arrays:
        array = np.asarray(source)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        if array.dtype.kind in {"O", "U", "S"}:
            for item in array.reshape(-1):
                encoded = str(item).encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "little"))
                digest.update(encoded)
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stratified_60_20_20(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(len(y))
    train, temporary = train_test_split(
        index, test_size=0.4, stratify=y, random_state=seed
    )
    validation, test = train_test_split(
        temporary, test_size=0.5, stratify=y[temporary], random_state=seed
    )
    return train, validation, test


def tabular_manifests() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = ["breast", "spambase", "mushroom", "adult", "bank"]
    split_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    mechanism_cs = {
        "scar": [0.1, 0.3, 0.5, 0.7],
        "sar_indep": [0.3, 0.5],
        "sar_attr": [0.3, 0.5],
        "sar_gap": [0.3, 0.5],
        "sar_extreme": [0.3, 0.5],
    }
    for dataset in datasets:
        for seed in range(5):
            x, y = load_dataset(dataset, max_rows=8000, seed=seed)
            train, validation, test = stratified_60_20_20(y, seed)
            split_rows.append({
                "track_family": "shared_tabular_60_20_20",
                "dataset": dataset,
                "seed": seed,
                "n_total": len(y),
                "n_features": x.shape[1],
                "n_train": len(train),
                "n_validation": len(validation),
                "n_test": len(test),
                "dataset_xy_sha256": array_hash(x, y),
                "train_order_sha256": array_hash(train),
                "validation_order_sha256": array_hash(validation),
                "test_order_sha256": array_hash(test),
                "split_recipe": "sklearn train_test_split; stratified 60/20/20; random_state=seed twice",
            })
            mu, sd = x[train].mean(0), x[train].std(0) + 1e-9
            x_train = (x[train] - mu) / sd
            y_train = y[train]
            x_validation = (x[validation] - mu) / sd
            y_validation = y[validation]
            for mechanism, cs in mechanism_cs.items():
                for c in cs:
                    train_pu = make_pu(x_train, y_train, c, mechanism, seed)
                    validation_pu = make_pu(
                        x_validation, y_validation, c, mechanism, seed + 1000
                    )
                    draw_rows.append({
                        "dataset": dataset,
                        "mechanism": mechanism,
                        "c_requested": c,
                        "split_seed": seed,
                        "train_draw_seed": seed,
                        "validation_draw_seed": seed + 1000,
                        "n_train_positive": int(y_train.sum()),
                        "n_train_labeled": len(train_pu.X_p),
                        "train_c_realized": f"{train_pu.c:.12g}",
                        "train_s_sha256": array_hash(train_pu.s),
                        "train_propensity_sha256": array_hash(train_pu.e_true),
                        "n_validation_positive": int(y_validation.sum()),
                        "n_validation_labeled": len(validation_pu.X_p),
                        "validation_c_realized": f"{validation_pu.c:.12g}",
                        "validation_s_sha256": array_hash(validation_pu.s),
                        "validation_propensity_sha256": array_hash(validation_pu.e_true),
                    })
    return split_rows, draw_rows


def representation_splits() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for representation in ["cifar_raw", "cifar_clip", "news_tfidf", "news_minilm"]:
        data = np.load(ROOT / "experiments" / "features" / f"{representation}.npz")
        x, y = data["X"], data["y"].astype(int)
        for seed in range(5):
            train, validation, test = stratified_60_20_20(y, seed)
            rows.append({
                "track_family": "frozen_representation_60_20_20",
                "dataset": representation,
                "seed": seed,
                "n_total": len(y),
                "n_features": x.shape[1],
                "n_train": len(train),
                "n_validation": len(validation),
                "n_test": len(test),
                "dataset_xy_sha256": array_hash(x, y),
                "train_order_sha256": array_hash(train),
                "validation_order_sha256": array_hash(validation),
                "test_order_sha256": array_hash(test),
                "split_recipe": "sklearn train_test_split; stratified 60/20/20; random_state=seed twice",
            })
    return rows


def static_native_splits() -> list[dict[str, Any]]:
    data = np.load(ROOT / "experiments" / "features" / "uniprot_nucleus.npz")
    x, y, s = data["X"], data["y"].astype(int), data["s"].astype(int)
    rows = []
    for seed in range(5):
        train, validation, test = stratified_60_20_20(y, seed)
        rows.append({
            "track_family": "static_native_swissprot_60_20_20",
            "dataset": "uniprot_nucleus",
            "seed": seed,
            "n_total": len(y),
            "n_features": x.shape[1],
            "n_train": len(train),
            "n_validation": len(validation),
            "n_test": len(test),
            "dataset_xy_sha256": array_hash(x, y, s),
            "train_order_sha256": array_hash(train),
            "validation_order_sha256": array_hash(validation),
            "test_order_sha256": array_hash(test),
            "split_recipe": "stratified 60/20/20 by y while preserving each row's observed s",
        })
    return rows


def graph_splits() -> list[dict[str, Any]]:
    from run_graph import load_graph, rewire

    rows = []
    for dataset in ["cora", "citeseer"]:
        x, multiclass_y, edges = load_graph(dataset)
        for seed in range(5):
            rng = np.random.default_rng(seed)
            permutation = rng.permutation(len(multiclass_y))
            n_train, n_validation = int(0.6 * len(permutation)), int(0.2 * len(permutation))
            train = permutation[:n_train]
            validation = permutation[n_train:n_train + n_validation]
            test = permutation[n_train + n_validation:]
            for fraction in [0.0, 0.4, 0.8]:
                changed_edges = rewire(edges, multiclass_y, fraction, seed)
                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "rewire_fraction": fraction,
                    "n_nodes": len(multiclass_y),
                    "n_edges": len(changed_edges),
                    "features_labels_edges_sha256": array_hash(x, multiclass_y, edges),
                    "rewired_edges_sha256": array_hash(changed_edges),
                    "train_order_sha256": array_hash(train),
                    "validation_order_sha256": array_hash(validation),
                    "test_order_sha256": array_hash(test),
                    "split_recipe": "numpy.default_rng(seed).permutation; first 60%, next 20%, final 20%",
                })
    return rows


def vision_draws() -> list[dict[str, Any]]:
    from run_end_to_end_vision_track import make_data

    rows = []
    root = ROOT / "experiments" / "features" / "end_to_end"
    for dataset in ["cifar10", "fashionmnist"]:
        for seed in [0, 1, 2]:
            p_data, u_data, _, test_data, y_u, backbone = make_data(
                dataset, root, seed, 0.2, 0
            )
            rows.append({
                "dataset": dataset,
                "seed": seed,
                "c": 0.2,
                "backbone": backbone,
                "n_labeled_positive": len(p_data.indices),
                "n_unlabeled": len(u_data.indices),
                "n_test": len(test_data.indices),
                "labeled_train_indices_sha256": array_hash(p_data.indices),
                "unlabeled_train_indices_sha256": array_hash(u_data.indices),
                "unlabeled_binary_labels_sha256": array_hash(y_u),
                "test_indices_labels_sha256": array_hash(
                    test_data.indices, test_data.binary_targets
                ),
                "draw_recipe": "SCAR c=0.2 independently within each of four positive classes; numpy.default_rng(seed)",
            })
    return rows


def temporal_cohort_masks() -> list[dict[str, Any]]:
    rows = []
    base = ROOT / "experiments" / "features" / "native_temporal"
    for dataset, file_name, id_name in [
        ("debian", "debian_vulnerability_temporal.npz", "packages"),
        ("swissprot", "swissprot_nucleus_temporal.npz", "accessions"),
    ]:
        data = np.load(base / file_name, allow_pickle=True)
        labels = data["labels"].astype(bool)
        present = data["present"].astype(bool)
        ids = data[id_name].astype(str)
        p0, validation, test = labels[:, 0], labels[:, 1], labels[:, 2]
        if dataset == "debian":
            known_by = data["known_by"].astype(bool)
            validation_control = present[:, 1] & (~known_by[:, 1])
            test_mask = present[:, 2] & (~known_by[:, 1])
        else:
            validation_control = (~p0) & (~validation) & present[:, 1]
            test_mask = (~p0) & (~validation) & present[:, 2]
        rows.append({
            "dataset": dataset,
            "model_seeds": "0|1|2|3|4",
            "n_entities": len(ids),
            "n_t0_positive": int(p0.sum()),
            "n_validation_event": int(validation.sum()),
            "n_validation_control": int(validation_control.sum()),
            "n_test_risk_set": int(test_mask.sum()),
            "n_test_event": int((test & test_mask).sum()),
            "entity_order_sha256": array_hash(ids),
            "t0_positive_mask_sha256": array_hash(p0),
            "validation_event_mask_sha256": array_hash(validation),
            "validation_control_mask_sha256": array_hash(validation_control),
            "test_risk_mask_sha256": array_hash(test_mask),
            "test_event_mask_sha256": array_hash(test & test_mask),
            "protocol": "experiments/features/native_temporal/experiment_protocol.json",
        })
    return rows


def main() -> None:
    split_rows, pu_draw_rows = tabular_manifests()
    split_rows.extend(representation_splits())
    split_rows.extend(static_native_splits())
    graph_rows = graph_splits()
    vision_rows = vision_draws()
    temporal_rows = temporal_cohort_masks()

    split_fields = [
        "track_family", "dataset", "seed", "n_total", "n_features", "n_train",
        "n_validation", "n_test", "dataset_xy_sha256", "train_order_sha256",
        "validation_order_sha256", "test_order_sha256", "split_recipe",
    ]
    write_csv(GENERATED / "split_seed_manifest.csv", split_rows, split_fields)
    write_csv(GENERATED / "tabular_pu_draw_manifest.csv", pu_draw_rows, [
        "dataset", "mechanism", "c_requested", "split_seed", "train_draw_seed",
        "validation_draw_seed", "n_train_positive", "n_train_labeled",
        "train_c_realized", "train_s_sha256", "train_propensity_sha256",
        "n_validation_positive", "n_validation_labeled", "validation_c_realized",
        "validation_s_sha256", "validation_propensity_sha256",
    ])
    write_csv(GENERATED / "graph_split_rewire_manifest.csv", graph_rows, [
        "dataset", "seed", "rewire_fraction", "n_nodes", "n_edges",
        "features_labels_edges_sha256", "rewired_edges_sha256",
        "train_order_sha256", "validation_order_sha256", "test_order_sha256",
        "split_recipe",
    ])
    write_csv(GENERATED / "vision_pu_draw_manifest.csv", vision_rows, [
        "dataset", "seed", "c", "backbone", "n_labeled_positive",
        "n_unlabeled", "n_test", "labeled_train_indices_sha256",
        "unlabeled_train_indices_sha256", "unlabeled_binary_labels_sha256",
        "test_indices_labels_sha256", "draw_recipe",
    ])
    write_csv(GENERATED / "temporal_cohort_mask_manifest.csv", temporal_rows, [
        "dataset", "model_seeds", "n_entities", "n_t0_positive",
        "n_validation_event", "n_validation_control", "n_test_risk_set",
        "n_test_event", "entity_order_sha256", "t0_positive_mask_sha256",
        "validation_event_mask_sha256", "validation_control_mask_sha256",
        "test_risk_mask_sha256", "test_event_mask_sha256", "protocol",
    ])
    report = {
        "status": "verified",
        "split_rows": len(split_rows),
        "tabular_pu_draw_rows": len(pu_draw_rows),
        "graph_split_rewire_rows": len(graph_rows),
        "vision_pu_draw_rows": len(vision_rows),
        "temporal_cohort_rows": len(temporal_rows),
    }
    (GENERATED / "split_seed_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
