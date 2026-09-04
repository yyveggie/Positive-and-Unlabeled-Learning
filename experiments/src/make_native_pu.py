from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(HERE, "features")
OUT = os.path.join(CACHE, "uniprot_nucleus.npz")
RAW = os.path.join(CACHE, "uniprot_human_raw.tsv")

URL = ("https://rest.uniprot.org/uniprotkb/stream"
       "?query=reviewed:true+AND+organism_id:9606"
       "&fields=accession,length,sequence,cc_subcellular_location"
       "&format=tsv")

AA = list("ACDEFGHIKLMNPQRSTVWY")
BASIC = set("KR")
TARGET = "Nucleus"
EXPERIMENTAL = "ECO:0000269"


def download(force=False):
    if os.path.exists(RAW) and not force:
        return RAW
    os.makedirs(CACHE, exist_ok=True)
    print("downloading Swiss-Prot human entries ...", flush=True)
    with urllib.request.urlopen(URL, timeout=600) as r, open(RAW, "wb") as fh:
        fh.write(r.read())
    print(f"  -> {RAW} ({os.path.getsize(RAW) / 1e6:.1f} MB)")
    return RAW


def parse_location(text):
    if not text or not text.strip():
        return None
    body = text.split("Note=")[0]
    body = body.replace("SUBCELLULAR LOCATION:", " ")
    nuclear = experimental = False
    for stmt in re.split(r"\.\s", body):
        if TARGET in stmt:
            nuclear = True
            if EXPERIMENTAL in stmt:
                experimental = True
    return nuclear, experimental


def featurize(seq):
    seq = seq.strip().upper()
    n = len(seq)
    counts = np.array([seq.count(a) for a in AA], dtype=np.float64)
    comp = counts / max(n, 1)

    w, best = 30, 0.0
    if n >= w:
        run = sum(ch in BASIC for ch in seq[:w])
        best = run
        for i in range(w, n):
            run += (seq[i] in BASIC) - (seq[i - w] in BASIC)
            best = max(best, run)
        best /= w
    elif n:
        best = sum(ch in BASIC for ch in seq) / n
    return np.concatenate([comp, [np.log10(max(n, 1)), best]])


def build():
    path = download()
    X, y, s, kept, skipped = [], [], [], 0, 0
    with open(path, encoding="utf8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i_seq, i_loc = header.index("Sequence"), len(header) - 1
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= i_loc:
                skipped += 1
                continue
            loc = parse_location(parts[i_loc])
            seq = parts[i_seq]
            if loc is None or not seq or set(seq) - set(AA):
                skipped += 1
                continue
            nuclear, experimental = loc
            X.append(featurize(seq))
            y.append(int(nuclear))
            s.append(int(nuclear and experimental))
            kept += 1
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=int)
    np.savez_compressed(OUT, X=X, y=y, s=s,
                        feature_names=np.array(AA + ["log_length", "basic30"]))
    print(f"kept {kept} entries, skipped {skipped}")
    report()


def report():
    d = np.load(OUT, allow_pickle=True)
    X, y, s = d["X"], d["y"], d["s"]
    alpha = y.mean()
    c = s.sum() / max(y.sum(), 1)
    print(f"\n{OUT}")
    print(f"  n = {len(y)}, d = {X.shape[1]}")
    print(f"  annotated nuclear (y=1)          {y.sum():6d}   alpha = {alpha:.4f}")
    print(f"  experimentally confirmed (s=1)   {s.sum():6d}   c     = {c:.4f}")
    print(f"  hidden positives in U            {(y & (1 - s)).sum():6d}")
    print(f"  labeled but not annotated        {(s & (1 - y)).sum():6d}"
          "   (should be 0 by construction)")


    ln = X[:, AA.__len__()]
    p_len, u_pos_len = ln[s == 1].mean(), ln[(y == 1) & (s == 0)].mean()
    print(f"  mean log10 length: labeled P {p_len:.3f} vs hidden positives "
          f"{u_pos_len:.3f}   difference {p_len - u_pos_len:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        build()
