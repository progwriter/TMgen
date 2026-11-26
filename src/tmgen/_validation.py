"""Argument checks shared by the models, the metrics and TrafficMatrix."""

from typing import Any

import numpy
from numpy.typing import ArrayLike, NDArray

from tmgen.exceptions import TMgenException


def count(value: float, name: str) -> int:
    """A positive whole number, such as a number of nodes or epochs."""
    try:
        whole = int(value)
    except (TypeError, ValueError, OverflowError):
        raise TMgenException(
            f"{name} must be a positive integer, got {value!r}"
        ) from None
    if whole != value or whole < 1:
        raise TMgenException(f"{name} must be a positive integer, got {value!r}")
    return whole


def number(
    value: float,
    name: str,
    *,
    minimum: float | None = 0,
    maximum: float | None = None,
    positive: bool = False,
) -> float:
    """A finite number, at least *minimum* (any if None), at most *maximum*."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise TMgenException(f"{name} must be a number, got {value!r}") from None
    if not numpy.isfinite(result):
        raise TMgenException(f"{name} must be finite, got {value!r}")
    if positive and result <= 0:
        raise TMgenException(f"{name} must be positive, got {value!r}")
    if minimum is not None and result < minimum:
        raise TMgenException(
            f"{name} must be finite and at least {minimum:g}, got {value!r}"
        )
    if maximum is not None and result > maximum:
        raise TMgenException(f"{name} must be at most {maximum:g}, got {value!r}")
    return result


def nonnegative(values: ArrayLike, name: str) -> NDArray[numpy.float64]:
    """An array of finite non-negative numbers."""
    try:
        array = numpy.asarray(values, dtype=numpy.float64)
    except (TypeError, ValueError):
        raise TMgenException(f"{name} must be an array of numbers") from None
    if not numpy.isfinite(array).all() or (array < 0).any():
        raise TMgenException(f"{name} must be finite and non-negative")
    return array


def series(values: ArrayLike, name: str) -> NDArray[numpy.float64]:
    """A non-empty one-dimensional array of finite non-negative numbers."""
    array = nonnegative(values, name)
    if array.ndim != 1 or array.size == 0:
        raise TMgenException(f"{name} must be a non-empty one-dimensional array")
    return array


def resolve_rng(rng: numpy.random.Generator | int | None) -> Any:
    """
    A source of random numbers with the methods of ``numpy.random.Generator``
    that the models use. ``None`` gives the ``numpy.random`` module itself, whose
    functions draw from NumPy's global state (and so follow ``numpy.random.seed``).
    """
    if rng is None:
        return numpy.random
    return numpy.random.default_rng(rng)
