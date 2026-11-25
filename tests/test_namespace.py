import subprocess
import sys

import pytest

import tmgen


def test_top_level_exports() -> None:
    assert issubclass(tmgen.TMgenException, ValueError)
    assert callable(tmgen.hmc.hmc_sample)
    assert callable(tmgen.models.exp_tm)
    assert isinstance(tmgen.__version__, str)
    for name in tmgen.__all__:
        if name != "plot":  # an optional dependency
            assert getattr(tmgen, name) is not None
    pytest.importorskip("matplotlib")
    assert tmgen.plot.heatmap is not None


def test_unknown_attribute() -> None:
    with pytest.raises(AttributeError, match="no_such_name"):
        tmgen.no_such_name  # noqa: B018


def test_plot_is_not_imported_with_the_package() -> None:
    code = "import sys, tmgen; print('tmgen.plot' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_dir_lists_only_the_public_names() -> None:
    assert dir(tmgen) == sorted(tmgen.__all__)
