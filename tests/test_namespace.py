import subprocess
import sys

import pytest

import tmgen


def test_public_names_exist() -> None:
    for name in tmgen.__all__:
        assert getattr(tmgen, name) is not None, name


def test_top_level_exports() -> None:
    assert issubclass(tmgen.TMgenException, ValueError)
    assert callable(tmgen.hmc.hmc_sample)
    assert callable(tmgen.models.exp_tm)
    assert isinstance(tmgen.__version__, str)


def test_unknown_attribute() -> None:
    with pytest.raises(AttributeError, match="no_such_name"):
        tmgen.no_such_name  # noqa: B018


def test_plot_is_not_imported_with_the_package() -> None:
    code = "import sys, tmgen; print('tmgen.plot' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_plot_is_available_on_demand() -> None:
    pytest.importorskip("matplotlib")
    assert tmgen.plot.heatmap is not None


def test_dir_lists_only_the_public_names() -> None:
    assert dir(tmgen) == sorted(tmgen.__all__)
