"""Bounded recapture without relaxing any D52 observation acceptance check."""

from collections.abc import Callable


def retry_capture[T](
    capture: Callable[[], T], on_retry: Callable[[int, str], None], attempts: int = 3
) -> T:
    """Recompute the same waypoint through the original capture implementation."""
    for attempt in range(1, attempts + 1):
        try:
            return capture()
        except ValueError as error:
            if not str(error).startswith(
                ("Actual camera missed policy goal by ", "Image/pose mismatch ")
            ):
                raise
            if attempt == attempts:
                raise
            on_retry(attempt, str(error))
    raise AssertionError("No capture attempted")
