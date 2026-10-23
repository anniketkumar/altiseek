"""CSV round-trips of dataclass tables."""

from dataclasses import dataclass
from datetime import datetime

import pytest

from altiseek.data.frames import Frame
from altiseek.data.tables import read_csv, write_csv


@dataclass(frozen=True)
class Row:
    name: str
    count: int
    value_m: float
    flag: bool
    when: datetime | None
    note: str | None
    ratio: float | None


ROWS = [
    Row("a", 1, 0.1 + 0.2, True, datetime(2020, 8, 25, 14, 16, 21, 467133), 'x, "quoted"', 1e-12),
    Row("", -3, 35.199999999999996, False, None, None, None),
]


def test_round_trip_preserves_values_and_none(tmp_path):
    write_csv(tmp_path / "t.csv", Row, ROWS)
    assert read_csv(tmp_path / "t.csv", Row) == ROWS


def test_empty_table_round_trips(tmp_path):
    write_csv(tmp_path / "t.csv", Row, [])
    assert read_csv(tmp_path / "t.csv", Row) == []


def test_column_mismatch_is_an_error(tmp_path):
    write_csv(tmp_path / "t.csv", Row, ROWS)
    with pytest.raises(ValueError, match="columns"):
        read_csv(tmp_path / "t.csv", Frame)


def test_unsupported_column_type_is_an_error(tmp_path):
    @dataclass(frozen=True)
    class Bad:
        items: list

    write_csv(tmp_path / "t.csv", Bad, [Bad([1])])
    with pytest.raises(TypeError, match="unsupported"):
        read_csv(tmp_path / "t.csv", Bad)
