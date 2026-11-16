"""Build the bug dashboard into `_site/` for GitHub Pages.

Copies `site/index.html` and writes `_site/bugs.json` with every record of `bugs/`,
its name, the file and line of its Lean refutation,
and its reproducer split into code and stated outputs.
Fails if the reproducer of a bug that is not fixed does not show the bug on the installed JAX.
Run it with `uv run python scripts/site.py` and preview it with `python -m http.server -d _site`.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from bugs import parse, reproduce

ROOT = Path(__file__).parent.parent
OUT = ROOT / "_site"


def git(*args: str) -> str | None:
    """Return the output of a git command in the repository, or None if it fails."""
    run = subprocess.run(
        ["git", "-C", str(ROOT), *args], check=False, capture_output=True, text=True
    )
    return run.stdout.strip() or None if run.returncode == 0 else None


def repository() -> str | None:
    """Return the GitHub URL of the repository, if known."""
    if "GITHUB_REPOSITORY" in os.environ:
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        return f"{server}/{os.environ['GITHUB_REPOSITORY']}"
    remote = git("remote", "get-url", "origin") or ""
    match = re.match(r"(?:https://github\.com/|git@github\.com:)(.+?)(?:\.git)?$", remote)
    return f"https://github.com/{match.group(1)}" if match else None


def locate(theorem: str) -> str | None:
    """Return `<file>:<line>` of the Lean theorem with the fully qualified name `theorem`."""
    short = re.escape(theorem.rsplit(".", 1)[-1])
    pattern = re.compile(rf"^\s*(?:private\s+)?theorem\s+{short}\b")
    for path in sorted((ROOT / "Jax").rglob("*.lean")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.match(line):
                return f"{path.relative_to(ROOT)}:{number}"
    return None


def installed(package: str) -> str | None:
    """Return the installed version of `package`, if any."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def record(path: Path) -> dict:
    """Return the bug record at `path`, with its name, Lean location and reproducer run."""
    bug = json.loads(path.read_text())
    bug.pop("$schema", None)
    bug["name"] = path.stem
    refutation = bug["lean_refutation"]
    bug["lean_location"] = locate(refutation) if refutation else None
    bug["parts"] = parse(bug["reproducer"])
    if bug["status"] != "fixed" and (errors := reproduce(bug)):
        sys.exit(f"site: {path.name}: " + "\n".join(errors))
    print(f"site: {path.name}: {bug['status']}")
    return bug


def main() -> None:
    data = {
        "repository": repository(),
        "commit": os.environ.get("GITHUB_SHA") or git("rev-parse", "HEAD"),
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
        "jax_installed": installed("jax"),
        "bugs": [record(path) for path in sorted((ROOT / "bugs").glob("*.json"))],
    }
    OUT.mkdir(exist_ok=True)
    shutil.copy(ROOT / "site" / "index.html", OUT / "index.html")
    (OUT / "bugs.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"site: {len(data['bugs'])} bugs written to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
