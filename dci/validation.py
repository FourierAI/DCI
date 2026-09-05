"""Exact raw-string validation for both Flat and local DCI calls."""

from __future__ import annotations

from dataclasses import dataclass

VALIDATION_VERSION = "exact-raw-string-v1"


@dataclass(frozen=True)
class ValidatedPrediction:
    label: str | None
    status: str  # selected, none, or invalid


def validate_prediction(raw: str, candidates: list[str]) -> ValidatedPrediction:
    """Accept only an untouched response that exactly matches this call's labels.

    No whitespace trimming, wrapper removal, punctuation removal, case conversion,
    semantic remapping, or substring extraction is performed. ``None`` is the
    sole null response and is recognized only with that exact spelling.
    """
    if raw == "None":
        return ValidatedPrediction(None, "none")
    if raw in candidates:
        return ValidatedPrediction(raw, "selected")
    return ValidatedPrediction(None, "invalid")
