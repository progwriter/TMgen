import functools
import warnings
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def renamed_arguments(**old_to_new: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Accept the old keyword names of renamed parameters, with a
    ``DeprecationWarning``. Positional calls are unaffected.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            for old, new in old_to_new.items():
                if old not in kwargs:
                    continue
                if new in kwargs:
                    raise TypeError(
                        f"{func.__name__}() got both '{old}' (deprecated) and '{new}'"
                    )
                warnings.warn(
                    f"'{old}' is deprecated, use '{new}' instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        return wrapper

    return decorator
