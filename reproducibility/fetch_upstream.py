import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "reproducibility" / "upstream_sources.csv"
PATCHES = {
    "DEDPUL_TIcE": ROOT / "reproducibility" / "patches" / "dedpul_compatibility.patch",
    "SAR_PU": ROOT / "reproducibility" / "patches" / "sarpu_sklearn_compatibility.patch",
}


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, check=True, text=True,
                          capture_output=True).stdout.strip()


def fetch_git(row, force):
    dest = ROOT / row["local_path"]
    rev = row["pinned_revision"]
    if dest.exists():
        if not force:
            if not (dest / ".git").exists():
                return "present (no git metadata; re-run with --force to verify the revision)"
            head = run(["git", "rev-parse", "HEAD"], cwd=dest)
            state = "ok" if head == rev else f"HEAD {head[:12]} != pinned {rev[:12]}"
            return f"present ({state})"
        subprocess.run(["rm", "-rf", str(dest)], check=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--quiet", row["upstream_url"], str(dest)])
    run(["git", "checkout", "--quiet", rev], cwd=dest)
    patch = PATCHES.get(row["component"])
    if patch and patch.exists():
        run(["git", "apply", str(patch)], cwd=dest)
        return f"cloned at {rev[:12]}, patch applied"
    return f"cloned at {rev[:12]}"


def check_hash(row):
    dest = ROOT / row["local_path"]
    name = row["revision_kind"].removeprefix("sha256_of_")
    target = dest / name
    if not target.exists():
        return (f"MISSING. Download {name} from {row['upstream_url']} into "
                f"{row['local_path']}/ and re-run to verify.")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return "ok" if digest.startswith(row["pinned_revision"][:32]) else f"SHA-256 mismatch: {digest[:32]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    failures = 0
    for row in csv.DictReader(SOURCES.open(encoding="utf-8")):
        try:
            if row["revision_kind"] == "git_commit":
                status = fetch_git(row, args.force)
            else:
                status = check_hash(row)
        except subprocess.CalledProcessError as exc:
            status = f"FAILED: {(exc.stderr or '').strip().splitlines()[-1:] or exc}"
        if "FAILED" in status or "MISSING" in status or "mismatch" in status or "!=" in status:
            failures += 1
        print(f"{row['component']:13} {status}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
