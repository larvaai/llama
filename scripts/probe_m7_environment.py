from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def module_probe(name: str) -> dict[str, Any]:
    available = importlib.util.find_spec(name) is not None
    version = None
    if available:
        try:
            version = importlib.metadata.version(name.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"available": available, "version": version}


def endpoint_probe(base_url: str | None) -> dict[str, Any]:
    if base_url is None:
        return {"status": "not_configured"}
    url = base_url.rstrip("/") + "/v1/models"
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            return {"status": "invalid", "detail": "response_too_large"}
        payload = json.loads(raw.decode("utf-8"))
        if type(payload) is not dict:
            return {"status": "invalid", "detail": "non_object_response"}
        return {
            "status": "reachable",
            "url": url,
            "model_count": len(payload.get("data", []))
            if type(payload.get("data")) is list
            else None,
        }
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {"status": "unreachable", "url": url, "detail": type(exc).__name__}


def command_probe(name: str) -> dict[str, Any]:
    executable = shutil.which(name)
    return {"available": executable is not None, "path": executable}


def wsl_probe() -> dict[str, Any]:
    executable = shutil.which("wsl")
    if executable is None:
        return {"available": False, "distributions": []}
    try:
        encoded = subprocess.check_output(
            [executable, "--list", "--quiet"],
            timeout=10,
        )
        raw = encoded.decode("utf-16-le", errors="replace")
        distributions = [line.strip("\x00 ") for line in raw.splitlines() if line.strip("\x00 ")]
    except (OSError, subprocess.SubprocessError):
        distributions = []
    return {"available": True, "distributions": distributions}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe optional M7 backend availability")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--vllm-url")
    parser.add_argument("--sglang-url")
    args = parser.parse_args()

    modules = {
        name: module_probe(name)
        for name in ("mlx", "mlx_lm", "vllm", "sglang")
    }
    providers = {
        "mlx_lm": {
            "status": "available_not_executed"
            if modules["mlx_lm"]["available"]
            else "unavailable",
            "reason": (
                "no model path was supplied to this environment probe"
                if modules["mlx_lm"]["available"]
                else "mlx-lm module is unavailable in this Python environment"
            ),
        },
        "vllm": endpoint_probe(args.vllm_url),
        "sglang": endpoint_probe(args.sglang_url),
    }
    artifact = {
        "artifact_version": "inference-runtime-m7-environment.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "commands": {
            "docker": command_probe("docker"),
            "wsl": wsl_probe(),
        },
        "python_modules": modules,
        "providers": providers,
        "decision": {
            "portable_adapter_conformance_can_run": True,
            "real_mlx_execution_available": providers["mlx_lm"]["status"]
            == "available_not_executed",
            "real_vllm_server_reachable": providers["vllm"]["status"] == "reachable",
            "real_sglang_server_reachable": providers["sglang"]["status"]
            == "reachable",
            "note": (
                "Provider availability is environment evidence, not a reason to "
                "overstate adapter capabilities or portable conformance."
            ),
        },
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["decision"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
