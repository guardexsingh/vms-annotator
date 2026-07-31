"""Check public project artifacts for configured camera URL leakage."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .config import load_config


PUBLIC_ENTRIES = ("app", "config", "scripts", "web", "docs", "README.md", "logs")
URL_WITH_CREDENTIALS = re.compile(r"rtsp(?:s)?://[^\s/'\"]+@", re.IGNORECASE)


def scan(root: Path, config_path: Path) -> list[str]:
    config = load_config(config_path)
    camera_urls = {camera.url for camera in config.cameras if camera.enabled and camera.url}
    failures: set[str] = set()
    for entry in PUBLIC_ENTRIES:
        path = root / entry
        candidates = path.rglob("*") if path.is_dir() else (path,)
        for candidate in candidates:
            if not candidate.is_file() or "__pycache__" in candidate.parts:
                continue
            try:
                if candidate.stat().st_size > 10 * 1024 * 1024:
                    continue
                text = candidate.read_text(errors="ignore")
            except OSError:
                continue
            if any(url in text for url in camera_urls) or URL_WITH_CREDENTIALS.search(text):
                failures.add(str(candidate.relative_to(root)))
    return sorted(failures)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("config/cameras.yaml"))
    args = parser.parse_args()
    failures = scan(args.root.resolve(), args.config)
    if failures:
        print("Credential scan failed in:")
        for path in failures:
            print(path)
        return 1
    print("Credential scan: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
