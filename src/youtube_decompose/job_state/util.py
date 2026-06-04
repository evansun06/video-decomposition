from __future__ import annotations

import unicodedata
from collections.abc import Iterable


DEFAULT_MAX_NON_LATIN_TITLE_RATIO = 0.20

_TRUE_VALUES = {"true", "1", "yes", "y", "downloaded"}
_FALSE_VALUES = {"false", "0", "no", "n", "not downloaded"}


def parse_bool(value: str, *, field_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Unsupported boolean value for {field_name}: {value!r}")


def parse_duration_minutes(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def should_exclude_duration(
    duration_minutes: float,
    *,
    max_duration_minutes: float,
) -> bool:
    return duration_minutes >= max_duration_minutes


def title_letter_counts(title: str) -> tuple[int, int]:
    normalized = unicodedata.normalize("NFKC", title)
    total_letters = 0
    non_latin_letters = 0

    for character in normalized:
        if not unicodedata.category(character).startswith("L"):
            continue

        total_letters += 1
        if "LATIN" not in unicodedata.name(character, ""):
            non_latin_letters += 1

    return total_letters, non_latin_letters


def non_latin_title_ratio(title: str) -> float:
    total_letters, non_latin_letters = title_letter_counts(title)
    if total_letters == 0:
        return 0.0
    return non_latin_letters / total_letters


def max_non_latin_title_ratio(titles: Iterable[str]) -> float:
    return max((non_latin_title_ratio(title) for title in titles), default=0.0)


def should_exclude_title(
    title: str,
    *,
    max_non_latin_title_ratio: float = DEFAULT_MAX_NON_LATIN_TITLE_RATIO,
) -> bool:
    return non_latin_title_ratio(title) > max_non_latin_title_ratio
