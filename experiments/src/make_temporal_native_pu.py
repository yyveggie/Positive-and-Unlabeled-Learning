from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import re
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE / "features" / "native_temporal"
RAW_DEBIAN = ROOT / "raw" / "debian"
RAW_UNIPROT = ROOT / "raw" / "uniprot"
OUT_DEBIAN = ROOT / "debian_vulnerability_temporal.npz"
OUT_UNIPROT = ROOT / "swissprot_nucleus_temporal.npz"
PROVENANCE = ROOT / "provenance.json"

HASH_DIM = 256
SNAPSHOT_DATES = ("2023-01-01", "2024-01-01", "2025-01-01", "2026-01-01")
AA = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA)
NUCLEAR_EXPERIMENT = re.compile(
    r"\bNucleus\b[^.;]*?\{[^}]*\bECO:0000269\b[^}]*\}"
)


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str
    size: int
    sha256: str
    commit: str = ""


@dataclass(frozen=True)
class UniProtFile:
    name: str
    url: str
    size: int
    md5: str
    release: str
    member: str = ""


@dataclass(frozen=True)
class ProteinRecord:
    primary: str
    accessions: tuple[str, ...]
    sequence: str
    sequence_sha256: str
    annotated_nuclear: bool
    experimental_nuclear: bool


DEBIAN_SOURCE_FILES = (
    RemoteFile(
        "debian_sources_2023-01-01.gz",
        "https://snapshot.debian.org/archive/debian/20230101T000000Z/"
        "dists/bullseye/main/source/Sources.gz",
        11_429_086,
        "006a5628b8afa45bf77cc449afa6f98647573b0e98b119a5944e65741094bdeb",
    ),
    RemoteFile(
        "debian_sources_2024-01-01.gz",
        "https://snapshot.debian.org/archive/debian/20240101T000000Z/"
        "dists/bullseye/main/source/Sources.gz",
        11_429_912,
        "10b8bc80b9bc4a9ae3bceb2d39cf3120774edc2f8e5693dbfa5d1445ff1de4af",
    ),
    RemoteFile(
        "debian_sources_2025-01-01.gz",
        "https://snapshot.debian.org/archive/debian/20250101T000000Z/"
        "dists/bullseye/main/source/Sources.gz",
        11_430_349,
        "52a771dd16da85821aa6fb402dcba71b8c27f41890defe9b140942de9eebb18d",
    ),
    RemoteFile(
        "debian_sources_2026-01-01.gz",
        "https://snapshot.debian.org/archive/debian/20260101T000000Z/"
        "dists/bullseye/main/source/Sources.gz",
        11_430_349,
        "52a771dd16da85821aa6fb402dcba71b8c27f41890defe9b140942de9eebb18d",
    ),
)

DEBIAN_TRACKER_FILES = (
    RemoteFile(
        "tracker_2023-01-01.txt",
        "https://salsa.debian.org/security-tracker-team/security-tracker/-/raw/"
        "e961d2fba09b732782f648435b5b29e35641adf7/data/CVE/list",
        31_917_106,
        "ab1cce6e44fc53368372b827adfb45c9022d48cffca401830c0d23669bfb39c3",
        "e961d2fba09b732782f648435b5b29e35641adf7",
    ),
    RemoteFile(
        "tracker_2024-01-01.txt",
        "https://salsa.debian.org/security-tracker-team/security-tracker/-/raw/"
        "984a4c0fa2871aacbc88cbc281cb3567a9595fd1/data/CVE/list",
        36_118_401,
        "57b3c226459ee7313d58ba4f2801f3de96df5fa8f750629d6a04e54719e09292",
        "984a4c0fa2871aacbc88cbc281cb3567a9595fd1",
    ),
    RemoteFile(
        "tracker_2025-01-01.txt",
        "https://salsa.debian.org/security-tracker-team/security-tracker/-/raw/"
        "01be7118d912146e402df7be63b7126a53ac2a6c/data/CVE/list",
        42_264_672,
        "d5b817e44f0f4d76ffb0e3c9d059073596a51de6935de20bed72fe9654acd12d",
        "01be7118d912146e402df7be63b7126a53ac2a6c",
    ),
    RemoteFile(
        "tracker_2026-01-01.txt",
        "https://salsa.debian.org/security-tracker-team/security-tracker/-/raw/"
        "ae1ac9d8607ffbcb9cd646215ca8ab6af258a946/data/CVE/list",
        49_814_854,
        "4e7820dad91fe88763835d42f6a64a62617a098a3d22afd1ff5981ff9cc6da5f",
        "ae1ac9d8607ffbcb9cd646215ca8ab6af258a946",
    ),
)

DEBIAN_FILES = DEBIAN_SOURCE_FILES + DEBIAN_TRACKER_FILES

UNIPROT_FILES = (
    UniProtFile(
        "uniprot_sprot-only2024_01.tar.gz",
        "https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/"
        "release-2024_01/knowledgebase/uniprot_sprot-only2024_01.tar.gz",
        1_610_292_503,
        "4af0782a896836489454b3f61979d360",
        "2024_01",
        "uniprot_sprot.dat.gz",
    ),
    UniProtFile(
        "uniprot_sprot-only2025_01.tar.gz",
        "https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/"
        "release-2025_01/knowledgebase/uniprot_sprot-only2025_01.tar.gz",
        1_688_438_192,
        "9b8760eb76329e619e175a3a55b9de33",
        "2025_01",
        "uniprot_sprot.dat.gz",
    ),
    UniProtFile(
        "uniprot_sprot_human_2026_02.dat.gz",
        "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
        "knowledgebase/taxonomic_divisions/uniprot_sprot_human.dat.gz",
        119_315_692,
        "89a29126d82f44b5a93e158a01e99c8b",
        "2026_02",
    ),
)


def file_sha256(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def file_md5(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def download_locked(spec: RemoteFile, retries: int = 8) -> Path:
    RAW_DEBIAN.mkdir(parents=True, exist_ok=True)
    path = RAW_DEBIAN / spec.name
    if path.exists() and path.stat().st_size == spec.size:
        got = file_sha256(path)
        if got == spec.sha256:
            print(f"verified {spec.name}: {spec.size:,} bytes")
            return path
        raise RuntimeError(f"hash mismatch for complete file {path}: {got}")
    if path.exists() and path.stat().st_size > spec.size:
        raise RuntimeError(f"oversized partial file; refusing to overwrite {path}")

    while not path.exists() or path.stat().st_size < spec.size:
        start = path.stat().st_size if path.exists() else 0
        stop = min(start + 4 * (1 << 20) - 1, spec.size - 1)
        req = urllib.request.Request(
            spec.url,
            headers={"Range": f"bytes={start}-{stop}", "Accept-Encoding": "identity"},
        )
        last_error = None
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as response:
                    status = getattr(response, "status", None)
                    body = response.read()
                    content_range = response.headers.get("Content-Range", "")
                if start and status != 206:
                    raise RuntimeError(f"resume returned HTTP {status}, not 206")
                if status == 206 and not content_range.startswith(f"bytes {start}-"):
                    raise RuntimeError(f"unexpected Content-Range {content_range!r}")
                if not body:
                    raise RuntimeError("empty response body")
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 == retries:
                    raise
                time.sleep(min(2 ** attempt, 20))
        if last_error and not body:
            raise last_error
        mode = "ab" if start else "wb"
        with path.open(mode) as fh:
            fh.write(body[: spec.size - start])
        now = path.stat().st_size
        print(f"downloaded {spec.name}: {now:,}/{spec.size:,}", flush=True)

    got = file_sha256(path)
    if got != spec.sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}: {got} != {spec.sha256}")
    print(f"verified {spec.name}: sha256={got}")
    return path


def download_debian() -> None:
    for spec in DEBIAN_FILES:
        download_locked(spec)


def download_uniprot() -> None:
    RAW_UNIPROT.mkdir(parents=True, exist_ok=True)
    for spec in UNIPROT_FILES:
        path = RAW_UNIPROT / spec.name
        if path.exists() and path.stat().st_size == spec.size:
            got = file_md5(path)
            if got != spec.md5:
                raise RuntimeError(f"MD5 mismatch for complete file {path}: {got}")
            print(f"verified {spec.name}: md5={got}")
            continue
        if path.exists() and path.stat().st_size > spec.size:
            raise RuntimeError(f"oversized partial file; refusing to overwrite {path}")
        while not path.exists() or path.stat().st_size < spec.size:
            start = path.stat().st_size if path.exists() else 0
            req = urllib.request.Request(
                spec.url,
                headers={"Range": f"bytes={start}-", "Accept-Encoding": "identity"},
            )
            with urllib.request.urlopen(req, timeout=600) as response:
                status = getattr(response, "status", None)
                if start and status != 206:
                    raise RuntimeError(f"resume returned HTTP {status}, not 206")
                mode = "ab" if start else "wb"
                with path.open(mode) as fh:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        remaining = spec.size - fh.tell()
                        if remaining <= 0:
                            break
                        fh.write(chunk[:remaining])
            print(f"downloaded {spec.name}: {path.stat().st_size:,}/{spec.size:,}")
        got = file_md5(path)
        if got != spec.md5:
            raise RuntimeError(f"MD5 mismatch for {path}: {got} != {spec.md5}")
        print(f"verified {spec.name}: md5={got}")


def _open_uniprot_flat(spec: UniProtFile):
    path = RAW_UNIPROT / spec.name
    if spec.member:
        archive = tarfile.open(path, mode="r:gz")
        member = archive.extractfile(spec.member)
        if member is None:
            archive.close()
            raise FileNotFoundError(f"{spec.member} is absent from {path}")
        compressed = gzip.GzipFile(fileobj=member, mode="rb")
        text_stream = io.TextIOWrapper(compressed, encoding="utf8", errors="replace")

        text_stream._uniprot_archive = archive
        text_stream._uniprot_member = member
        return text_stream
    return gzip.open(path, "rt", encoding="utf8", errors="replace")


def parse_uniprot_release(spec: UniProtFile) -> dict[str, ProteinRecord]:
    records: dict[str, ProteinRecord] = {}
    stream = _open_uniprot_flat(spec)
    accessions: list[str] = []
    sequence_parts: list[str] = []
    location_parts: list[str] = []
    is_human = in_sequence = in_location = False

    def finish() -> None:
        nonlocal accessions, sequence_parts, location_parts
        if is_human and accessions and sequence_parts:
            sequence = "".join(sequence_parts).upper()
            location = " ".join(location_parts)
            primary = accessions[0]
            records[primary] = ProteinRecord(
                primary=primary,
                accessions=tuple(dict.fromkeys(accessions)),
                sequence=sequence,
                sequence_sha256=hashlib.sha256(sequence.encode("ascii", errors="ignore")).hexdigest(),
                annotated_nuclear=bool(re.search(r"\bNucleus\b", location)),
                experimental_nuclear=bool(NUCLEAR_EXPERIMENT.search(location)),
            )

    try:
        for line in stream:
            if line.startswith("//"):
                finish()
                accessions, sequence_parts, location_parts = [], [], []
                is_human = in_sequence = in_location = False
                continue
            if line.startswith("AC   "):
                accessions.extend(x.strip() for x in line[5:].split(";") if x.strip())
            elif line.startswith("OX   ") and re.search(r"NCBI_TaxID=9606(?:;|\b)", line):
                is_human = True
            elif line.startswith("CC   -!- SUBCELLULAR LOCATION:"):
                in_location = True
                location_parts.append(line.split("SUBCELLULAR LOCATION:", 1)[1].strip())
            elif in_location and line.startswith("CC       "):
                location_parts.append(line[9:].strip())
            elif in_location:
                in_location = False
            if line.startswith("SQ   "):
                in_sequence = True
            elif in_sequence and line.startswith("     ") and is_human:
                sequence_parts.append("".join(line.split()))
    finally:
        stream.close()
        archive = getattr(stream, "_uniprot_archive", None)
        if archive is not None:
            archive.close()
    print(f"UniProt {spec.release}: {len(records):,} reviewed human entries; "
          f"experimental nucleus={sum(r.experimental_nuclear for r in records.values()):,}; "
          f"any-source nucleus={sum(r.annotated_nuclear for r in records.values()):,}")
    return records


def uniprot_features(sequence: str) -> np.ndarray:
    n = len(sequence)
    composition = np.array([sequence.count(a) for a in AA], dtype=np.float32) / max(n, 1)
    width, best = 30, 0
    if n:
        initial = min(width, n)
        run = sum(ch in "KR" for ch in sequence[:initial])
        best = run
        for i in range(width, n):
            run += int(sequence[i] in "KR") - int(sequence[i - width] in "KR")
            best = max(best, run)
    basic = best / min(width, max(n, 1))
    nonstandard = sum(ch not in AA_SET for ch in sequence) / max(n, 1)
    return np.concatenate([
        composition,
        np.array([math.log10(max(n, 1)), basic, nonstandard], dtype=np.float32),
    ])


def _accession_index(records: dict[str, ProteinRecord]) -> tuple[dict[str, ProteinRecord], int]:


    index: dict[str, ProteinRecord] = dict(records)
    secondary: dict[str, ProteinRecord] = {}
    ambiguous: set[str] = set()
    for record in records.values():
        for accession in record.accessions[1:]:
            if accession in index:
                continue
            if accession in secondary and secondary[accession].primary != record.primary:
                ambiguous.add(accession)
            else:
                secondary[accession] = record
    for accession in ambiguous:
        secondary.pop(accession, None)
    index.update(secondary)
    return index, len(ambiguous)


def _read_provenance() -> dict:
    if not PROVENANCE.exists():
        return {}
    return json.loads(PROVENANCE.read_text(encoding="utf8"))


def _write_provenance(provenance: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with PROVENANCE.open("w", encoding="utf8") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=True)
        fh.write("\n")


def build_uniprot() -> None:
    for spec in UNIPROT_FILES:
        path = RAW_UNIPROT / spec.name
        if not path.exists():
            raise FileNotFoundError(f"run --download-uniprot first; missing {path}")
        if path.stat().st_size != spec.size or file_md5(path) != spec.md5:
            raise RuntimeError(f"raw-file integrity check failed: {path}")

    snapshots = [parse_uniprot_release(spec) for spec in UNIPROT_FILES]
    base = snapshots[0]
    later_indices, alias_collisions = [], []
    for records in snapshots[1:]:
        index, collisions = _accession_index(records)
        later_indices.append(index)
        alias_collisions.append(collisions)

    accessions = np.array(sorted(base), dtype=str)
    X = np.stack([uniprot_features(base[a].sequence) for a in accessions])
    n = len(accessions)
    present = np.ones((n, 3), dtype=np.int8)
    annotated = np.zeros((n, 3), dtype=np.int8)
    experimental = np.zeros((n, 3), dtype=np.int8)
    sequence_changed = np.zeros((n, 3), dtype=np.int8)
    later_records: list[list[ProteinRecord | None]] = [[], []]
    for i, accession in enumerate(accessions):
        first = base[accession]
        annotated[i, 0] = first.annotated_nuclear
        experimental[i, 0] = first.experimental_nuclear
        for j, index in enumerate(later_indices, start=1):
            record = index.get(accession)
            later_records[j - 1].append(record)
            present[i, j] = int(record is not None)
            if record is not None:
                annotated[i, j] = record.annotated_nuclear
                experimental[i, j] = record.experimental_nuclear
                sequence_changed[i, j] = record.sequence_sha256 != first.sequence_sha256

    labels = np.zeros((n, 3), dtype=np.int8)
    labels[:, 0] = experimental[:, 0]
    labels[:, 1] = ((experimental[:, 0] == 0) & (experimental[:, 1] == 1)).astype(np.int8)
    labels[:, 2] = ((experimental[:, 0] == 0) & (experimental[:, 1] == 0)
                    & (experimental[:, 2] == 1)).astype(np.int8)
    feature_names = np.array(list(AA) + ["log_length", "basic30", "nonstandard_fraction"])
    np.savez_compressed(
        OUT_UNIPROT,
        X=X,
        accessions=accessions,
        labels=labels,
        present=present,
        annotated_nuclear=annotated,
        experimental_nuclear=experimental,
        sequence_changed=sequence_changed,
        feature_names=feature_names,
        label_names=np.array(["confirmed_by_t0", "new_t0_t1", "new_t1_t2"]),
        releases=np.array([spec.release for spec in UNIPROT_FILES]),
    )

    p0 = labels[:, 0].astype(bool)
    val = labels[:, 1].astype(bool)
    test = labels[:, 2].astype(bool)
    val_risk = (~p0) & present[:, 1].astype(bool)
    test_risk = (~p0) & (~val) & present[:, 2].astype(bool)
    aliases_rescued = [
        sum(record is not None and record.primary != accession
            for accession, record in zip(accessions, records))
        for records in later_records
    ]
    counts = {
        "universe_t0": n,
        "confirmed_by_t0": int(p0.sum()),
        "unlabeled_at_t0": int((~p0).sum()),
        "present_t1": int(present[:, 1].sum()),
        "validation_risk_set": int(val_risk.sum()),
        "new_t0_t1": int(val.sum()),
        "present_t2": int(present[:, 2].sum()),
        "test_risk_set": int(test_risk.sum()),
        "new_t1_t2": int(test.sum()),
        "still_unconfirmed_controls_t2": int(test_risk.sum() - test.sum()),
        "any_source_nuclear_per_snapshot": [int(annotated[:, j].sum()) for j in range(3)],
        "experimental_nuclear_per_snapshot_in_t0_cohort": [int(experimental[:, j].sum()) for j in range(3)],
        "t0_sequences_changed_by_snapshot": [int(sequence_changed[:, j].sum()) for j in range(3)],
        "secondary_accessions_rescuing_t0_entities": aliases_rescued,
        "alias_collisions_in_later_snapshots": alias_collisions,
    }
    provenance = _read_provenance()
    provenance["swissprot"] = {
        "task": "first future experimental nucleus-location confirmation",
        "cohort": "all reviewed human proteins present in UniProtKB/Swiss-Prot 2024_01",
        "boundaries": [spec.release for spec in UNIPROT_FILES],
        "raw_files": [spec.__dict__ for spec in UNIPROT_FILES],
        "positive_rule": "Nucleus location with ECO:0000269 in the same location statement",
        "entity_linkage": "later primary accession preferred; otherwise a unique secondary accession; ambiguous aliases censored",
        "feature_policy": "2024_01 amino-acid sequence only, frozen for all later outcomes",
        "control_semantics": "present but not experimentally confirmed by the stated horizon; not a verified negative",
        "prior_semantics": "latent biological class prior unknown; observed confirmation rates are lower bounds",
        "counts": counts,
        "derived_file": str(OUT_UNIPROT.relative_to(HERE)),
        "derived_sha256": file_sha256(OUT_UNIPROT),
    }
    _write_provenance(provenance)
    print(json.dumps(counts, indent=2))
    print(f"wrote {OUT_UNIPROT}")
    print(f"updated {PROVENANCE}")


def parse_deb822(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf8", errors="replace") as fh:
        fields: dict[str, str] = {}
        key = None
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                if fields:
                    yield fields
                fields, key = {}, None
            elif line[0].isspace() and key:
                fields[key] += "\n" + line.strip()
            elif ":" in line:
                key, value = line.split(":", 1)
                fields[key] = value.strip()
        if fields:
            yield fields


def _count_csv(value: str) -> int:
    return sum(bool(x.strip()) for x in value.split(","))


def _source_bytes(files_field: str) -> int:
    total = 0
    for line in files_field.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-2].isdigit():
            total += int(parts[-2])
    return total


def _dependency_names(value: str) -> list[str]:
    names = []
    for item in re.split(r"[,|]", value):
        match = re.search(r"[A-Za-z0-9][A-Za-z0-9+.-]+", item)
        if match:
            names.append(match.group(0).lower())
    return names


def _email_domain(value: str) -> str:
    match = re.search(r"<[^<>@]+@([^<>]+)>", value)
    return match.group(1).lower() if match else "unknown"


def _hash_token(vector: np.ndarray, token: str) -> None:
    digest = hashlib.blake2b(token.encode("utf8"), digest_size=8).digest()
    code = int.from_bytes(digest, "little")
    index = code % HASH_DIM
    sign = 1.0 if (code >> 63) == 0 else -1.0
    vector[index] += sign


def debian_features(fields: dict[str, str]) -> np.ndarray:
    deps = _dependency_names(fields.get("Build-Depends", ""))
    standards = fields.get("Standards-Version", "0")
    match = re.match(r"(\d+)(?:\.(\d+))?", standards)
    standards_value = (float(match.group(1)) + float(match.group(2) or 0) / 10
                       if match else 0.0)
    numeric = np.array([
        math.log1p(_source_bytes(fields.get("Files", ""))),
        math.log1p(len(deps)),
        math.log1p(_count_csv(fields.get("Binary", ""))),
        math.log1p(len(fields.get("Architecture", "").split())),
        math.log1p(_count_csv(fields.get("Uploaders", ""))) if fields.get("Uploaders") else 0.0,
        standards_value,
        float("Vcs-Git" in fields or "Vcs-Browser" in fields),
        float("Testsuite" in fields),
        float("Homepage" in fields),
    ], dtype=np.float32)
    hashed = np.zeros(HASH_DIM, dtype=np.float32)
    for dep in deps:
        _hash_token(hashed, f"dep={dep}")
    for arch in fields.get("Architecture", "").split():
        _hash_token(hashed, f"arch={arch.lower()}")
    for key in ("Section", "Priority", "Format"):
        _hash_token(hashed, f"{key.lower()}={fields.get(key, 'unknown').lower()}")
    _hash_token(hashed, f"maintainer_domain={_email_domain(fields.get('Maintainer', ''))}")
    return np.concatenate([numeric, hashed])


def parse_sources(path: Path) -> tuple[np.ndarray, np.ndarray]:
    records: dict[str, list[np.ndarray]] = {}
    duplicates = 0
    for fields in parse_deb822(path):
        name = fields.get("Package", "").strip()
        if not name:
            continue
        if name in records:
            duplicates += 1
        records.setdefault(name, []).append(debian_features(fields))
    packages = np.array(sorted(records), dtype=str)


    X = np.stack([np.mean(records[p], axis=0) for p in packages])
    print(f"Debian universe: {len(packages):,} source packages; "
          f"{duplicates} extra version stanzas aggregated; d={X.shape[1]}")
    return packages, X


def parse_source_names(path: Path) -> set[str]:
    return {
        name
        for fields in parse_deb822(path)
        if (name := fields.get("Package", "").strip())
    }


PACKAGE_LINE = re.compile(r"^\t(?:\[[^]]+\]\s+)?-\s+(\S+)\s+(.+)$")
EXCLUDED_STATUSES = ("<not-affected>", "<itp>", "<undetermined>")


def parse_tracker(path: Path) -> tuple[set[str], dict[str, int]]:
    counts: dict[str, int] = {}
    in_cve = False
    seen_this_cve: set[str] = set()
    with path.open(encoding="utf8", errors="replace") as fh:
        for line in fh:
            if line.startswith("CVE-"):
                in_cve = True
                seen_this_cve = set()
                continue
            if line and not line[0].isspace():
                in_cve = False
                continue
            if not in_cve:
                continue
            match = PACKAGE_LINE.match(line.rstrip("\n"))
            if not match:
                continue
            package, status = match.groups()
            if package in seen_this_cve or status.startswith(EXCLUDED_STATUSES):
                continue
            seen_this_cve.add(package)
            counts[package] = counts.get(package, 0) + 1
    return set(counts), counts


def build_debian() -> None:
    missing = [spec.name for spec in DEBIAN_FILES
               if not (RAW_DEBIAN / spec.name).exists()]
    if missing:
        raise FileNotFoundError(f"run --download-debian first; missing {missing}")
    for spec in DEBIAN_FILES:
        path = RAW_DEBIAN / spec.name
        if path.stat().st_size != spec.size or file_sha256(path) != spec.sha256:
            raise RuntimeError(f"raw-file integrity check failed: {path}")

    packages, X = parse_sources(RAW_DEBIAN / DEBIAN_SOURCE_FILES[0].name)
    universe = set(packages.tolist())
    present_sets = [
        parse_source_names(RAW_DEBIAN / spec.name) & universe
        for spec in DEBIAN_SOURCE_FILES
    ]
    tracker_sets, tracker_counts = [], []
    for spec in DEBIAN_TRACKER_FILES:
        assigned, counts = parse_tracker(RAW_DEBIAN / spec.name)
        tracker_sets.append(assigned & universe)
        tracker_counts.append(counts)


    cumulative_known: list[set[str]] = []
    ever_known: set[str] = set()
    interval_events: list[set[str]] = []
    for present, assigned in zip(present_sets, tracker_sets):
        newly_assigned = assigned - ever_known
        interval_events.append(newly_assigned & present)
        ever_known |= assigned
        cumulative_known.append(set(ever_known))

    p0, val, test, audit = interval_events
    labels = np.zeros((len(packages), 4), dtype=np.int8)
    for j, group in enumerate((p0, val, test, audit)):
        labels[:, j] = np.fromiter((p in group for p in packages), dtype=np.int8)
    present = np.zeros((len(packages), 4), dtype=np.int8)
    known_by = np.zeros((len(packages), 4), dtype=np.int8)
    for j in range(4):
        present[:, j] = np.fromiter(
            (p in present_sets[j] for p in packages), dtype=np.int8
        )
        known_by[:, j] = np.fromiter(
            (p in cumulative_known[j] for p in packages), dtype=np.int8
        )

    feature_names = np.array([
        "log_source_bytes", "log_build_dep_count", "log_binary_count",
        "log_architecture_count", "log_uploader_count", "standards_version",
        "has_vcs", "has_testsuite", "has_homepage",
    ] + [f"hash_{i:03d}" for i in range(HASH_DIM)])
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DEBIAN,
        X=X,
        packages=packages,
        labels=labels,
        present=present,
        known_by=known_by,
        feature_names=feature_names,
        label_names=np.array(["confirmed_by_t0", "new_t0_t1", "new_t1_t2", "new_t2_t3"]),
        snapshot_dates=np.array(SNAPSHOT_DATES),
    )

    nonmonotone = [len(tracker_sets[i] - tracker_sets[i + 1])
                   for i in range(len(tracker_sets) - 1)]
    risk_sets = [
        present_sets[j] - (cumulative_known[j - 1] if j else set())
        for j in range(4)
    ]
    controls = [present_sets[j] - cumulative_known[j] for j in range(4)]
    assigned_while_absent = [
        len((tracker_sets[j] - (cumulative_known[j - 1] if j else set()))
            - present_sets[j])
        for j in range(4)
    ]
    counts = {
        "universe": len(packages),
        "confirmed_by_t0": len(p0),
        "unlabeled_at_t0": len(packages) - len(p0),
        "present_per_snapshot": [len(x) for x in present_sets],
        "validation_risk_set": len(risk_sets[1]),
        "new_t0_t1": len(val),
        "validation_controls": len(controls[1]),
        "test_risk_set": len(risk_sets[2]),
        "new_t1_t2": len(test),
        "test_controls": len(controls[2]),
        "audit_risk_set": len(risk_sets[3]),
        "new_t2_t3": len(audit),
        "audit_controls": len(controls[3]),
        "mapped_packages_per_snapshot": [len(x) for x in tracker_sets],
        "packages_removed_from_next_snapshot": nonmonotone,
        "new_assignments_censored_for_snapshot_absence": assigned_while_absent,
        "tracker_cve_assignments_per_snapshot": [sum(c.values()) for c in tracker_counts],
    }
    provenance = _read_provenance()
    provenance["debian"] = {
            "task": "first future Debian Security Tracker CVE assignment",
            "universe_snapshot": "Debian bullseye main source packages at 2023-01-01",
            "boundaries": list(SNAPSHOT_DATES),
            "raw_files": [spec.__dict__ for spec in DEBIAN_FILES],
            "feature_policy": "t0 Sources metadata only; duplicate package versions averaged; no vulnerability-history feature",
            "status_policy": "top-level CVE package assignments except <not-affected>, <itp>, and <undetermined>",
            "presence_policy": "an entity enters a boundary-specific risk set only if present in the corresponding bullseye Sources snapshot; absent entities are censored",
            "known_status_policy": "a tracker assignment remains known at later boundaries even if an entry is subsequently removed",
            "control_semantics": "present at the boundary and not confirmed by that horizon; not a verified negative",
            "counts": counts,
            "derived_file": str(OUT_DEBIAN.relative_to(HERE)),
            "derived_sha256": file_sha256(OUT_DEBIAN),
    }
    _write_provenance(provenance)
    print(json.dumps(counts, indent=2))
    print(f"wrote {OUT_DEBIAN}")
    print(f"wrote {PROVENANCE}")


def report() -> None:
    if not PROVENANCE.exists():
        raise FileNotFoundError(f"not built: {PROVENANCE}")
    print(PROVENANCE.read_text(encoding="utf8"))


def audit() -> None:
    provenance = _read_provenance()
    checks = {}
    for name, path, id_key in (
        ("debian", OUT_DEBIAN, "packages"),
        ("swissprot", OUT_UNIPROT, "accessions"),
    ):
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True)
        X = data["X"]
        ids = data[id_key].astype(str)
        labels = data["labels"].astype(bool)
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{name}: duplicate entity IDs")
        if not np.isfinite(X).all():
            raise RuntimeError(f"{name}: non-finite feature value")
        if np.any(labels.sum(axis=1) > 1):
            raise RuntimeError(f"{name}: temporal confirmation intervals overlap")
        if "present" in data.files:
            present = data["present"].astype(bool)
            if np.any(labels[:, 1:] & ~present[:, 1:]):
                raise RuntimeError(f"{name}: an event is assigned while the entity is absent")
        if name == "debian":
            known_by = data["known_by"].astype(bool)
            if np.any(known_by[:, 1:] < known_by[:, :-1]):
                raise RuntimeError("debian: cumulative known status is nonmonotone")
            if np.any(labels & ~known_by):
                raise RuntimeError("debian: an event is not represented in known status")
        expected_hash = provenance[name]["derived_sha256"]
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"{name}: derived hash mismatch")
        checks[name] = {
            "entities_unique": len(ids),
            "features_finite": list(X.shape),
            "nonoverlapping_interval_labels": labels.sum(axis=0).astype(int).tolist(),
            "derived_sha256_verified": actual_hash,
        }
    print(json.dumps(checks, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-debian", action="store_true")
    parser.add_argument("--build-debian", action="store_true")
    parser.add_argument("--download-uniprot", action="store_true")
    parser.add_argument("--build-uniprot", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if not (args.download_debian or args.build_debian or args.download_uniprot
            or args.build_uniprot or args.report or args.audit):
        parser.error("choose at least one action")
    if args.download_debian:
        download_debian()
    if args.build_debian:
        build_debian()
    if args.download_uniprot:
        download_uniprot()
    if args.build_uniprot:
        build_uniprot()
    if args.report:
        report()
    if args.audit:
        audit()


if __name__ == "__main__":
    main()
