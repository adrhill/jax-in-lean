"""Check the bug records in `bugs/` and run their reproducers on the installed JAX.

A reproducer asserts the wrong behaviour, so it must succeed unless the bug is marked as fixed.
Pass `--show <name>` to print the reproducer of `bugs/<name>.json` instead.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCHEMA = json.loads((ROOT / "scripts" / "bug.schema.json").read_text())
STATUSES = SCHEMA["properties"]["status"]["enum"]


def check(bug: dict) -> list[str]:
    """Return the ways in which `bug` violates the schema."""
    errors = [f"missing key {key!r}" for key in SCHEMA["required"] if key not in bug]
    errors += [f"unknown key {key!r}" for key in bug if key not in SCHEMA["properties"]]
    if errors:
        return errors
    status = bug["status"]
    if status not in STATUSES:
        return [f"status must be one of {STATUSES}"]
    if status == "refuted" and bug["lean_refutation"] is None:
        errors.append(f"status {status!r} needs a lean_refutation")
    if status in ("reported", "fixed") and bug["upstream_issue"] is None:
        errors.append(f"status {status!r} needs an upstream_issue")
    if status == "fixed" and "fixed_in" not in bug:
        errors.append("status 'fixed' needs fixed_in")
    return errors


def main() -> int:
    if sys.argv[1:2] == ["--show"]:
        bug = json.loads((ROOT / "bugs" / f"{sys.argv[2]}.json").read_text())
        print("\n".join(bug["reproducer"]))
        return 0
    failed = False
    for path in sorted((ROOT / "bugs").glob("*.json")):
        bug = json.loads(path.read_text())
        errors = check(bug)
        if not errors and bug["status"] != "fixed":
            code = "\n".join(bug["reproducer"])
            run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            if run.returncode != 0:
                errors.append(f"the reproducer no longer shows the bug:\n{run.stderr}")
        for error in errors:
            print(f"{path.name}: {error}", file=sys.stderr)
        if not errors:
            print(f"{path.name}: {bug['status']}")
        failed |= bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
