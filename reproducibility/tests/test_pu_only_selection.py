#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "reproducibility" / "pu_only_selection.py"
spec = importlib.util.spec_from_file_location("pu_only_selection", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


class PUOnlySelectionTests(unittest.TestCase):
    def test_lee_liu_numeric_definition(self) -> None:
        score = module.lee_liu_from_scores([0.9, 0.8, 0.1], [0.9, 0.4, 0.2])
        self.assertTrue(math.isclose(score, 4.0 / 3.0))

    def test_oracle_or_outcome_metric_is_rejected(self) -> None:
        rows = [{"cell": "a", "crit_oraclef1": "1", "crit_leeliu": "0"}]
        with self.assertRaises(ValueError):
            module.select_configurations(rows, ["cell"], "crit_oraclef1")
        with self.assertRaises(ValueError):
            module.select_configurations(rows, ["cell"], "bacc")

    def test_selection_uses_only_declared_criterion_and_stable_tie_break(self) -> None:
        rows = [
            {"cell": "a", "lr": "0.01", "crit_leeliu": "0.8", "test_f1": "0.99"},
            {"cell": "a", "lr": "0.001", "crit_leeliu": "0.8", "test_f1": "0.01"},
            {"cell": "b", "lr": "0.01", "crit_leeliu": "0.4", "test_f1": "0.99"},
            {"cell": "b", "lr": "0.001", "crit_leeliu": "0.7", "test_f1": "0.01"},
        ]
        chosen = module.select_configurations(rows, ["cell"])
        self.assertEqual([(row["cell"], row["lr"]) for row in chosen],
                         [("a", "0.001"), ("b", "0.001")])

    def test_public_selection_signatures_have_no_label_argument(self) -> None:
        files_and_functions = {
            MODULE_PATH: {"lee_liu_from_scores", "select_configurations"},
            ROOT / "experiments" / "src" / "pu_lib.py": {
                "criteria_from_scores", "lee_liu", "all_criteria"
            },
        }
        forbidden = {"y", "y_true", "labels", "validation_labels", "test_labels"}
        for path, functions in files_and_functions.items():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            found = {node.name: node for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name in functions}
            self.assertEqual(set(found), functions)
            for name, node in found.items():
                arguments = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
                self.assertFalse(arguments & forbidden, f"{path}:{name} reads labels")


if __name__ == "__main__":
    unittest.main()
