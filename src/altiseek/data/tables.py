"""CSV storage for tables of frozen dataclass rows.

Cells are written with ``str`` (floats therefore round-trip exactly), datetimes as ISO 8601,
and ``None`` as an empty cell. Reading converts each column back using the dataclass field
types; supported types are ``int``, ``float``, ``str``, ``bool``, ``datetime`` and ``X | None``.
"""

import csv
import dataclasses
import types
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, get_args, get_type_hints


def write_csv(path: str | Path, row_type: type, rows: Iterable[Any]) -> None:
    """Write ``rows`` (instances of the dataclass ``row_type``) to ``path`` with a header."""
    names = [field.name for field in dataclasses.fields(row_type)]
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(names)
        for row in rows:
            writer.writerow(_cell(getattr(row, name)) for name in names)


def read_csv[T](path: str | Path, row_type: type[T]) -> list[T]:
    """Read a table written by :func:`write_csv` back into ``row_type`` instances."""
    hints = get_type_hints(row_type)
    names = [field.name for field in dataclasses.fields(row_type)]
    parsers = {name: _parser(hints[name]) for name in names}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != names:
            raise ValueError(f"{path}: columns {reader.fieldnames} do not match {names}")
        return [row_type(**{name: parsers[name](row[name]) for name in names}) for row in reader]


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _parser(field_type: Any) -> Callable[[str], Any]:
    args = get_args(field_type)
    if isinstance(field_type, types.UnionType) and type(None) in args:
        (inner,) = (arg for arg in args if arg is not type(None))
        parse = _parser(inner)
        return lambda cell: None if cell == "" else parse(cell)
    if field_type is datetime:
        return datetime.fromisoformat
    if field_type is bool:
        return {"True": True, "False": False}.__getitem__
    if field_type in (int, float, str):
        return field_type
    raise TypeError(f"unsupported column type {field_type!r}")
