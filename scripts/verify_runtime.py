"""Verify pinned runtime/model inputs before startup or native compilation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def default_runtime_library() -> str:
    """Platform-native shared library name produced by a llama.cpp build."""
    if sys.platform == "darwin":
        return "libllama.dylib"
    if sys.platform == "win32":
        return "llama.dll"
    return "libllama.so"


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
    runtime_data = data.get("runtime", {})
    runtime = Path(runtime_data.get("directory", ""))
    library_name = runtime_data.get("library", default_runtime_library())
    library = runtime / library_name
    library_sha = runtime_data.get("library_sha256", runtime_data.get("llama_dll_sha256"))
    for label, path, expected in (
        ("model", model, data.get("gguf_sha256")),
        (library_name, library, library_sha),
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
