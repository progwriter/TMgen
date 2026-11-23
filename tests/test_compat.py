import pytest

from tmgen._compat import renamed_arguments


@renamed_arguments(old="new")
def _func(first: int, new: int = 0) -> tuple[int, int]:
    return first, new


def test_new_name_does_not_warn(recwarn: pytest.WarningsRecorder) -> None:
    assert _func(1, new=2) == (1, 2)
    assert _func(1, 2) == (1, 2)
    assert not recwarn.list


def test_old_name_warns_and_is_forwarded() -> None:
    with pytest.warns(DeprecationWarning, match="'old' is deprecated, use 'new'"):
        assert _func(1, old=2) == (1, 2)  # type: ignore[call-arg]


def test_old_and_new_together_is_an_error() -> None:
    with pytest.raises(TypeError, match="both 'old'"):
        _func(1, old=2, new=3)  # type: ignore[call-arg]


def test_metadata_is_kept() -> None:
    assert _func.__name__ == "_func"
