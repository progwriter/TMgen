import datetime
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "release_notes", Path(__file__).parent.parent / "tools" / "release_notes.py"
)
assert _spec is not None and _spec.loader is not None
release_notes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_notes)

CHANGELOG = """# Changelog

## [Unreleased]

### Added

- New thing.

## [0.2.0] - 2026-01-01

### Fixed

- Old bug.

## [0.1.0] - 2025-01-01

### Added

- Empty-ish.
"""


def test_extract_returns_only_that_versions_notes() -> None:
    notes = release_notes.extract(CHANGELOG, "v0.2.0")
    assert notes == "### Fixed\n\n- Old bug."


def test_extract_unknown_version() -> None:
    with pytest.raises(release_notes.ChangelogError):
        release_notes.extract(CHANGELOG, "9.9.9")


def test_promote_moves_unreleased_entries_under_the_new_version() -> None:
    out = release_notes.promote(CHANGELOG, "0.3.0", datetime.date(2026, 2, 3))
    assert (
        "## [Unreleased]\n\n## [0.3.0] - 2026-02-03\n\n### Added\n\n- New thing." in out
    )
    assert release_notes.extract(out, "0.3.0") == "### Added\n\n- New thing."


def test_promote_rejects_empty_unreleased_and_duplicates() -> None:
    once = release_notes.promote(CHANGELOG, "0.3.0", datetime.date(2026, 2, 3))
    with pytest.raises(release_notes.ChangelogError):
        release_notes.promote(once, "0.4.0", datetime.date(2026, 2, 4))
    with pytest.raises(release_notes.ChangelogError):
        release_notes.promote(CHANGELOG, "0.2.0", datetime.date(2026, 2, 4))
