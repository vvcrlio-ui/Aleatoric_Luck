"""Strict design validation, before coercion or sampling."""

from numbers import Integral


def validate_size_grid(values, name: str, capacity: int | None = None) -> tuple[int, ...]:
    values = tuple(values)
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in values):
        raise ValueError(f"{name} must contain non-boolean positive integers")
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing without duplicates")
    if capacity is not None and values[-1] > capacity:
        raise ValueError(f"{name} exceeds actual capacity {capacity}: {values[-1]}")
    return tuple(int(v) for v in values)
