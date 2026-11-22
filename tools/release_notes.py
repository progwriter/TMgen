"""Helpers for CHANGELOG.md: extract a version's notes, promote Unreleased."""

import argparse
import datetime
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger("release_notes")

UNRELEASED = "## [Unreleased]"
_HEADER = re.compile(r"^## \[(?P<name>[^\]]+)\]", re.MULTILINE)


class ChangelogError(Exception):
    pass


def _section(text: str, name: str) -> str:
    headers = list(_HEADER.finditer(text))
    for i, match in enumerate(headers):
        if match["name"] == name:
            end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
            return text[match.end() : end].split("\n", 1)[-1].strip()
    raise ChangelogError(f"no section [{name}] in the changelog")


def _has_entries(body: str) -> bool:
    return any(line.startswith("- ") for line in body.splitlines())


def extract(text: str, version: str) -> str:
    """Return the notes of a released version; it must have entries."""
    body = _section(text, version.removeprefix("v"))
    if not _has_entries(body):
        raise ChangelogError(f"section [{version}] has no entries")
    return body


def promote(text: str, version: str, date: datetime.date) -> str:
    """Rename the Unreleased entries to a new version and start a fresh section."""
    version = version.removeprefix("v")
    if re.search(rf"^## \[{re.escape(version)}\]", text, re.MULTILINE):
        raise ChangelogError(f"version {version} is already in the changelog")
    if not _has_entries(_section(text, "Unreleased")):
        raise ChangelogError("nothing under [Unreleased] to release")
    return text.replace(
        UNRELEASED, f"{UNRELEASED}\n\n## [{version}] - {date.isoformat()}", 1
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    commands = parser.add_subparsers(dest="command", required=True)
    extract_cmd = commands.add_parser("extract", help="print a version's notes")
    extract_cmd.add_argument("--version", required=True)
    promote_cmd = commands.add_parser("promote", help="release the Unreleased notes")
    promote_cmd.add_argument("--version", required=True)
    promote_cmd.add_argument("--date", type=datetime.date.fromisoformat)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    text = args.changelog.read_text()
    try:
        if args.command == "extract":
            sys.stdout.write(extract(text, args.version) + "\n")
        else:
            date = args.date or datetime.date.today()
            args.changelog.write_text(promote(text, args.version, date))
            log.info("Promoted [Unreleased] to %s", args.version)
    except ChangelogError as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
