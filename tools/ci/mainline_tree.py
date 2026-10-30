"""CI tripwire (decision D4): unpack the newest mainline kernel into DEST and
apply our local kernel patch commits on top. Prints the mainline version.

Usage: python tools/ci/mainline_tree.py DEST
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

RELEASES = "https://www.kernel.org/releases.json"
REPO = Path(__file__).resolve().parents[2]


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


def main(dest: Path) -> None:
    with urllib.request.urlopen(RELEASES, timeout=60) as r:
        releases = json.load(r)["releases"]
    mainline = next(r for r in releases if r["moniker"] == "mainline")
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar") as tar:
        with urllib.request.urlopen(mainline["source"], timeout=600) as r:
            while chunk := r.read(1 << 20):
                tar.write(chunk)
        tar.flush()
        subprocess.run(["tar", "-xf", tar.name, "-C", str(dest), "--strip-components=1"], check=True)
    imported = git("log", "--diff-filter=A", "--format=%H", "--", "third_party/linux/Makefile").split()[-1]
    for commit in git("rev-list", "--reverse", f"{imported}..HEAD", "--", "third_party/linux").split():
        patch = git("format-patch", "-1", "--stdout", commit, "--", "third_party/linux")
        # a/third_party/linux/x -> x: strip three leading components.
        subprocess.run(["patch", "-p3", "--forward", "-d", str(dest)], input=patch, text=True, check=True)
    print(mainline["version"])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
