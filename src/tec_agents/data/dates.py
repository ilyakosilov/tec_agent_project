"""Date parsing helpers using the project-wide [start, end) convention."""

from __future__ import annotations

import calendar
import re
from datetime import date


MONTHS: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def parse_month_range(month_name: str, year: int | str) -> tuple[str, str]:
    """Return the half-open date interval covering one calendar month."""

    month_key = month_name.strip().lower()
    if month_key not in MONTHS:
        raise ValueError(f"Unknown month name: {month_name!r}")

    year_int = int(year)
    month = MONTHS[month_key]
    start = date(year_int, month, 1)

    if month == 12:
        end = date(year_int + 1, 1, 1)
    else:
        end = date(year_int, month + 1, 1)

    return start.isoformat(), end.isoformat()


def parse_date_range_from_text(text: str) -> tuple[str, str]:
    """
    Extract a date interval from text using the [start, end) convention.

    Explicit ISO ranges are treated as already half-open:
    "from 2024-03-10 to 2024-03-20" -> ("2024-03-10", "2024-03-20").
    Month references map to the first day of the next month as exclusive end.
    """

    lower = text.lower()

    explicit = re.search(
        r"(?:from\s+)?(\d{4}-\d{2}-\d{2})\s+(?:to|through|-)\s+(\d{4}-\d{2}-\d{2})",
        lower,
    )
    if explicit:
        start, end = explicit.group(1), explicit.group(2)
        _validate_iso_interval(start, end)
        return start, end

    month_match = re.search(
        rf"\b({'|'.join(re.escape(name) for name in sorted(MONTHS, key=len, reverse=True))})\b"
        r"\s+"
        r"\b(20\d{2}|19\d{2})\b",
        lower,
    )
    if month_match:
        return parse_month_range(month_match.group(1), month_match.group(2))

    # Also accept "2024 March".
    reverse_month_match = re.search(
        r"\b(20\d{2}|19\d{2})\b"
        r"\s+"
        rf"\b({'|'.join(re.escape(name) for name in sorted(MONTHS, key=len, reverse=True))})\b",
        lower,
    )
    if reverse_month_match:
        return parse_month_range(reverse_month_match.group(2), reverse_month_match.group(1))

    raise ValueError("Could not find an explicit date range or month/year in text")


def expected_hourly_points(start: str, end: str) -> int | None:
    """Return expected hourly sample count for a half-open interval."""

    start_date = date.fromisoformat(start[:10])
    end_date = date.fromisoformat(end[:10])
    days = (end_date - start_date).days
    if days < 0:
        return None
    return days * 24


def month_end_inclusive_to_exclusive(value: str) -> str | None:
    """
    Convert a last-calendar-day date to the next day if it is month-end.

    This is used only for diagnostics/corrections, not as an automatic tool call
    rewrite.
    """

    parsed = date.fromisoformat(value[:10])
    last_day = calendar.monthrange(parsed.year, parsed.month)[1]
    if parsed.day != last_day:
        return None

    if parsed.month == 12:
        return date(parsed.year + 1, 1, 1).isoformat()
    return date(parsed.year, parsed.month + 1, 1).isoformat()


def _validate_iso_interval(start: str, end: str) -> None:
    """Raise if a half-open ISO date interval is invalid."""

    if date.fromisoformat(end[:10]) <= date.fromisoformat(start[:10]):
        raise ValueError(f"end must be greater than start: start={start}, end={end}")


__all__ = [
    "MONTHS",
    "expected_hourly_points",
    "month_end_inclusive_to_exclusive",
    "parse_date_range_from_text",
    "parse_month_range",
]
