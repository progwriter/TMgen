from importlib.metadata import version

import tmgen


def test_version_matches_package_metadata() -> None:
    assert tmgen.__version__ == version("tmgen")
    assert tmgen.__version__
