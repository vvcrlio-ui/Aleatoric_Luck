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


def select_grid_points(values, selection: str, name: str) -> tuple[int, ...]:
    """Select existing grid points; an even grid uses its upper middle point.

    Applying min/middle/max to an already frozen three-point grid is idempotent.
    """

    values = validate_size_grid(values, name)
    if selection == "all":
        return values
    if selection != "min_middle_max":
        raise ValueError(f"Unknown grid_selection: {selection!r}")
    if len(values) < 3:
        raise ValueError(f"{name} needs at least three production grid points for pilot")
    return values[0], values[len(values) // 2], values[-1]
