"""Check the bug records in `bugs/` and run their reproducers on the installed JAX.

A reproducer is a plain script that prints what JAX computes.
A comment below each `print` states the output:
`# output: <text>` where JAX is correct,
and `# desired: <text>` followed by `# actual: <text>` where it is not.
Text of several lines continues in comments that are indented to the same column.
`parse` reads these comments, and `reproduce` checks them against what the script prints,
so a reproducer must print its `actual` outputs on the JAX version of its record.
Pass `--show <name>` to print the reproducer of `bugs/<name>.json`.
Pass `--jax <version>` to run the reproducers on another JAX release instead, or `--jax latest`:
a bug that no longer shows there is fixed or has changed, and its record needs an update.
"""

import difflib
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCHEMA = json.loads((ROOT / "scripts" / "bug.schema.json").read_text())
STATUSES = SCHEMA["properties"]["status"]["enum"]
PRIORITIES = SCHEMA["properties"]["priority"]["enum"]
SEARCH = SCHEMA["properties"]["duplicate_search"]["properties"]
QUERY = re.compile(SEARCH["queries"]["items"]["pattern"])
OUTPUT = re.compile(r"\s*# (output|desired|actual):\s+")
# Text that refers to Lean or to this repository, which only the field `lean` may contain.
INTERNAL = re.compile(r"\bLean\b|\bJax\.|\.lean\b|\bbugs/|\bscripts/|\b[Mm]odel(?:led|s)?\b")
STANDALONE = ("title", "description", "expected", "reproducer", "cause", "fix")


def check(bug: dict) -> list[str]:
    """Return the ways in which `bug` violates the schema."""
    errors = [f"missing key {key!r}" for key in SCHEMA["required"] if key not in bug]
    errors += [f"unknown key {key!r}" for key in bug if key not in SCHEMA["properties"]]
    if errors:
        return errors
    if list(bug) != [key for key in SCHEMA["properties"] if key in bug]:
        errors.append(f"keys must follow the order of the schema: {list(SCHEMA['properties'])}")
    for key in ("description", "expected", "reproducer", "cause", "lean"):
        if not bug[key]:
            errors.append(f"{key} must not be empty")
    if bug["fix"] == []:
        errors.append("fix must be null or not empty")
    for key in STANDALONE:
        text = bug[key] or []
        for line in [text] if isinstance(text, str) else text:
            if match := INTERNAL.search(line):
                errors.append(f"{key} refers to {match[0]!r}, which only lean may: {line}")
    if bug["priority"] not in PRIORITIES:
        errors.append(f"priority must be one of {PRIORITIES}")
    if bug["maybe_intended"] == []:
        errors.append("maybe_intended must be null or not empty")
    errors += check_search(bug)
    for name in bug.get("related", []):
        if not (ROOT / "bugs" / f"{name}.json").exists():
            errors.append(f"related bug {name!r} does not exist")
    status = bug["status"]
    if status not in STATUSES:
        return [f"status must be one of {STATUSES}"]
    if status == "proven" and bug["lean_refutation"] is None:
        errors.append(f"status {status!r} needs a lean_refutation")
    if status in ("reported", "fixed") and bug["upstream_issue"] is None:
        errors.append(f"status {status!r} needs an upstream_issue")
    if status == "fixed" and "fixed_in" not in bug:
        errors.append("status 'fixed' needs fixed_in")
    return errors


def check_search(bug: dict) -> list[str]:
    """Return the ways in which the duplicate search of `bug` violates the schema."""
    search = bug["duplicate_search"]
    if search is None:
        return (
            []
            if bug["upstream_issue"]
            else ["duplicate_search may be null only with an upstream_issue"]
        )
    if not isinstance(search, dict) or list(search) != list(SEARCH):
        return [f"duplicate_search must have the keys {list(SEARCH)}"]
    errors = []
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", search["date"]):
        errors.append("the date of duplicate_search must be YYYY-MM-DD")
    if not search["queries"]:
        errors.append("duplicate_search needs a query")
    errors += [
        f"query {query!r} must be `<owner>/<repository>: <query>`"
        for query in search["queries"]
        if not QUERY.fullmatch(query)
    ]
    return errors


def parse(reproducer: list[str]) -> list[str | dict[str, str]]:
    """Split a reproducer into its code and the outputs that its comments state.

    Code is a string, and an output is `{"desired": <text>, "actual": <text>}`,
    with equal texts for an `# output:` comment.
    """
    parts: list[str | dict[str, str]] = []
    code: list[str] = []
    desired = None
    i = 0
    while i < len(reproducer):
        match = OUTPUT.match(reproducer[i])
        if not match:
            code.append(reproducer[i])
            i += 1
            continue
        # A continuation line has spaces in place of the label: `#          [1. 3. 5.]]`.
        indent = re.sub(r"[^#\s]", " ", match[0])
        text = [reproducer[i][match.end() :]]
        i += 1
        while i < len(reproducer) and reproducer[i].startswith(indent):
            text.append(reproducer[i][len(indent) :])
            i += 1
        if match[1] == "desired":
            desired = "\n".join(text)
            continue
        actual = "\n".join(text)
        if (desired is None) != (match[1] == "output"):
            raise ValueError(f"`# actual: {text[0]}` needs a `# desired:` comment before it")
        if code:
            parts.append("\n".join(code))
            code = []
        parts.append({"desired": actual if desired is None else desired, "actual": actual})
        desired = None
    if desired is not None:
        raise ValueError(f"`# desired: {desired}` needs an `# actual:` comment after it")
    if code:
        parts.append("\n".join(code))
    return parts


def reproduce(bug: dict, python: tuple[str, ...] = (sys.executable,)) -> list[str]:
    """Run the reproducer of `bug` and return the ways in which it fails.

    `python` is the command of the interpreter, by default the one of the installed JAX.
    The reproducer must print the `actual` outputs of its comments, one of which differs from the desired one.
    """
    try:
        outputs = [part for part in parse(bug["reproducer"]) if isinstance(part, dict)]
    except ValueError as e:
        return [str(e)]
    if all(output["desired"] == output["actual"] for output in outputs):
        return ["the reproducer has no output with `# desired:` and a different `# actual:`"]
    code = "\n".join(bug["reproducer"])
    run = subprocess.run([*python, "-c", code], check=False, capture_output=True, text=True)
    if run.returncode != 0:
        return [f"the reproducer fails with exit code {run.returncode}:\n{run.stderr}"]
    stated = "\n".join(output["actual"] for output in outputs).splitlines()
    printed = run.stdout.splitlines()
    if printed != stated:
        diff = difflib.unified_diff(stated, printed, "comments", "printed", lineterm="")
        return ["the reproducer does not print its `actual` outputs:\n" + "\n".join(diff)]
    return []


def other_jax(version: str) -> int:
    """Run the reproducers on the JAX release `version`, or on the latest one, and report which still show their bug."""
    major, minor = sys.version_info[:2]
    package = "jax" if version == "latest" else f"jax=={version}"
    # `--isolated`: without it, `uv` reuses the environment of the pinned JAX.
    uv = f"uv run --no-project --isolated --python {major}.{minor} --with {package} python"
    python = tuple(uv.split())
    show = "import jax; print(jax.__version__)"
    run = subprocess.run([*python, "-c", show], check=False, capture_output=True, text=True)
    if run.returncode != 0:
        print(f"cannot run {package}:\n{run.stderr}", file=sys.stderr)
        return 1
    version = run.stdout.strip()
    print(f"jax {version}")
    release = lambda v: tuple(int(part) for part in v.split("."))
    changed = False
    for path in sorted((ROOT / "bugs").glob("*.json")):
        bug = json.loads(path.read_text())
        errors = reproduce(bug, python)
        if bug["status"] == "fixed" and release(version) >= release(bug["fixed_in"]):
            # The bug must be gone: the reproducer prints something else than its `actual` outputs.
            errors = (
                []
                if errors
                else [f"the reproducer still shows the bug, fixed in {bug['fixed_in']}"]
            )
        for error in errors:
            print(f"{path.name}: {error}", file=sys.stderr)
        if not errors:
            print(f"{path.name}: as recorded ({bug['status']})")
        changed |= bool(errors)
    return 1 if changed else 0


def main() -> int:
    if sys.argv[1:2] == ["--show"]:
        bug = json.loads((ROOT / "bugs" / f"{sys.argv[2]}.json").read_text())
        print("\n".join(bug["reproducer"]))
        return 0
    if sys.argv[1:2] == ["--jax"]:
        return other_jax(sys.argv[2])
    failed = False
    installed = importlib.metadata.version("jax")
    for path in sorted((ROOT / "bugs").glob("*.json")):
        bug = json.loads(path.read_text())
        errors = check(bug)
        # A fixed bug still shows on the JAX version on which it was found.
        if not errors and (bug["status"] != "fixed" or bug["jax_version"] == installed):
            errors = reproduce(bug)
        for error in errors:
            print(f"{path.name}: {error}", file=sys.stderr)
        if not errors:
            print(f"{path.name}: {bug['status']}")
        failed |= bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
