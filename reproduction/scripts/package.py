"""Create an upload archive from MANIFEST.json only, excluding runtime files."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

from verify import ROOT, verify


def package(output):
    verify()
    output = Path(output).resolve()
    if output.is_relative_to(ROOT.resolve()):
        raise ValueError("Write the archive outside the reproduction directory")
    if output.exists():
        raise FileExistsError("Archive already exists; choose a fresh output path")
    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    files = {r["path"]: r["sha256"] for r in manifest["files"]}
    files["MANIFEST.json"] = hashlib.sha256((ROOT / "MANIFEST.json").read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, expected in files.items():
            raw = (ROOT / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError(f"File changed during packaging: {name}")
            archive.writestr("reproduction/" + name, raw)
    with zipfile.ZipFile(output) as archive:
        if set(archive.namelist()) != {"reproduction/" + name for name in files} or archive.testzip() is not None:
            raise ValueError("Archive content/integrity mismatch")
        for name, expected in files.items():
            if hashlib.sha256(archive.read("reproduction/" + name)).hexdigest() != expected:
                raise ValueError(f"Archive byte mismatch: {name}")
    return {"archive": str(output), "files": len(files), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.output), indent=2))
