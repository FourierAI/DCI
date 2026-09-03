"""Deterministic whole-label validation for both Flat and local DCI calls."""

from __future__ import annotations

import re
from dataclasses import dataclass

VALIDATION_VERSION = "whole-label-v1"
_PREFIX = re.compile(r"^(?:answer\s*:|a\s*:|the answer is\s+)\s*", re.IGNORECASE)


@dataclass(frozen=True)
class ValidatedPrediction:
    label: str | None
    status: str  # selected, none, or invalid


def validate_prediction(raw: str, candidates: list[str]) -> ValidatedPrediction:
    """Normalize fixed wrappers, then match one complete local category name.

    Case is restored to the supplied candidate spelling. No substring search,
    semantic remapping, or comma splitting is performed: a comma-separated
    ImageNet synset description can itself be one complete category name.
    """
    candidate_map: dict[str, set[str]] = {}
    for label in candidates:
        candidate_map.setdefault(label.casefold(), set()).add(label)

    def resolve(value: str) -> ValidatedPrediction | None:
        if value.casefold() == "none":
            return ValidatedPrediction(None, "none")
        matches = candidate_map.get(value.casefold(), set())
        if len(matches) == 1:
            return ValidatedPrediction(next(iter(matches)), "selected")
        return None

    cleaned = raw.strip()
    result = resolve(cleaned)
    if result is not None:
        return result
    cleaned = _PREFIX.sub("", cleaned, count=1).strip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1].rstrip()
    for marker in ("```", "`", '"', "'"):
        if (
            len(cleaned) >= 2 * len(marker)
            and cleaned.startswith(marker)
            and cleaned.endswith(marker)
        ):
            cleaned = cleaned[len(marker) : -len(marker)].strip()
            break
    return resolve(cleaned) or ValidatedPrediction(None, "invalid")
