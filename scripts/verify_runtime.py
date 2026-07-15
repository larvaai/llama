"""Verify pinned runtime/model inputs before startup or native compilation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def verify(manifest_path: Path) -> list[str]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if data.get("runtime_build") != "b10012":
        errors.append("runtime_build must be b10012")
    model = Path(data.get("gguf_path", ""))
    runtime = Path(data.get("runtime", {}).get("directory", ""))
    dll = runtime / "llama.dll"
    for label, path, expected in (
        ("model", model, data.get("gguf_sha256")),
        ("llama.dll", dll, data.get("runtime", {}).get("llama_dll_sha256")),
    ):
        if not path.is_file():
            errors.append(f"{label} not found: {path}")
        elif not isinstance(expected, str) or not expected.startswith("sha256:"):
            errors.append(f"{label} digest must use sha256:<hex>")
        elif sha256(path) != expected.lower():
            errors.append(f"{label} SHA-256 mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    errors = verify(args.manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("runtime and model hashes verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
