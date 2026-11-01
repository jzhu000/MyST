"""Verify the standalone package, all fixed-ID assets, and optional fact files."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from myst_fusion.data import DATASETS, Dataset, digest, read_json, validate_rules, write_json
from myst_fusion.entity_types import type_frequencies
from myst_fusion.workflow import check_benchmark


def package_files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if (path.is_file() and relative.parts[0] not in ("outputs", "data", "processed", "models", "build", ".pytest_cache")
                and not {"__pycache__", "build", ".pytest_cache"}.intersection(relative.parts)
                and path.suffix not in (".pyc", ".so", ".pyd")
                and path.name != "MANIFEST.json"):
            yield path


def refresh_manifest():
    write_json(ROOT / "MANIFEST.json", {
        "date": "2026-10-31", "scope": "M7 implementation, final types, rule banks, evaluation scripts",
        "excludes": "MANIFEST.json itself; runtime data, caches, models, outputs and compiled binaries",
        "files": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)} for p in package_files()]})


def verify(data_root=None):
    if (ROOT / "assets/type_audit").exists() or (ROOT / "scripts/audit_types.py").exists():
        raise ValueError("Internal type-label reviews belong outside the M7 distribution")
    manifest = read_json(ROOT / "MANIFEST.json")
    listed = {row["path"] for row in manifest["files"]}
    actual = {p.relative_to(ROOT).as_posix() for p in package_files()}
    if listed != actual:
        raise ValueError(f"Package inventory changed: missing={listed - actual}, extra={actual - listed}")
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if digest(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Package file changed: {row['path']}")
    for row in read_json(ROOT / "assets/provenance.json")["files"]:
        if digest(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"Supplied asset bytes changed: {row['path']}")
    summary = {}
    for name in DATASETS:
        data = Dataset(ROOT, name, data_root or "data")
        counts = read_json(data.asset / "relation_type_counts.json")
        if set(counts) - set(data.relations.values()):
            raise ValueError("Unknown relation in type counts")
        for relation, pairs in counts.items():
            total = sum(pairs.values())
            if total <= 0 or any(type(n) is not int or n < 0 for n in pairs.values()):
                raise ValueError("Invalid training type counts")
            frequencies = data.patterns[relation]
            if set(frequencies) != set(pairs) or any(abs(frequencies[p] - n / total) > 1e-12 for p, n in pairs.items()):
                raise ValueError("Type frequencies/counts disagree")
        rule_counts = {}
        for path in sorted((data.asset / "rules").glob("*.json")):
            rules = read_json(path)
            validate_rules(rules, len(data.relations))
            rule_counts[path.stem] = sum(map(len, rules.values()))
        summary[name] = {"entities": len(data.entities), "relations": len(data.relations), "rules": rule_counts}
        if data_root:
            data.load_parts()
            for split, facts in data.parts.items():
                check_benchmark(ROOT, name, split, facts)
            actual_counts, _ = type_frequencies(data.parts["train"], data.types, data.relations)
            if actual_counts != counts:
                raise ValueError(f"Training-only pattern counts differ: {name}")
            summary[name]["benchmark_and_training_counts"] = "passed"
    return {"files": len(listed), "datasets": summary}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", help="Optional directory containing the five imported benchmark datasets")
    p.add_argument("--refresh-manifest", action="store_true", help="For maintainers after intentional package edits")
    args = p.parse_args()
    if args.refresh_manifest:
        refresh_manifest()
    print(json.dumps(verify(args.data_root), indent=2))
