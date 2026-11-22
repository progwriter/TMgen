import pytest


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    # doctests in docs/ write files (e.g. pickles) to the current directory
    monkeypatch.chdir(tmp_path)
