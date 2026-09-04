from __future__ import annotations

import argparse, csv, itertools, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from run_ext import FIELDS, HP_GRID, TABULAR, run_cell

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mnist", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    out = os.path.join(HERE, "results",
                       "dcpu_smoke.csv" if args.smoke
                       else "dcpu_mnist_results.csv" if args.mnist
                       else "dcpu_results.csv")
    new = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        if args.smoke:
            run_cell(w, fh, "smoke", "breast", 0.5, 0, ["dcpu"], [(1e-2, 0.0)])
        elif args.mnist:
            for c, seed in itertools.product([0.3, 0.5], args.seeds):
                print(f"[dcpu-mnist] c={c} seed={seed}", flush=True)
                run_cell(w, fh, "mnist", "mnist20k", c, seed, ["dcpu"],
                         [(1e-3, 0.0), (1e-4, 1e-4)], epochs=30, hidden=128,
                         max_rows=20000)
        else:
            for ds, c, seed in itertools.product(TABULAR, [0.1, 0.3, 0.5, 0.7],
                                                 args.seeds):
                print(f"[dcpu] {ds} c={c} seed={seed}", flush=True)
                run_cell(w, fh, "ext", ds, c, seed, ["dcpu"], HP_GRID)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
