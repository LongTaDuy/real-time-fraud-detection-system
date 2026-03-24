"""Normalize and validate list/filter query parameters."""

from __future__ import annotations

import re

from fastapi import HTTPException, status

from app.services.transaction_read_service import TransactionListFilters

_PREDICTED_LABEL_RE = re.compile(r"^[\w-]{1,32}$")


def optional_trimmed(value: str | None, *, max_length: int, field: str) -> str | None:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    if len(s) > max_length:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": f"{field} exceeds maximum length {max_length}"},
        )
    if "\x00" in s:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": f"{field} contains invalid characters"},
        )
    return s


def optional_predicted_label(value: str | None) -> str | None:
    s = optional_trimmed(value, max_length=32, field="predicted_label")
    if s is None:
        return None
    if not _PREDICTED_LABEL_RE.fullmatch(s):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "predicted_label must be 1-32 chars: letters, digits, underscore, hyphen",
            },
        )
    return s


def transaction_list_filters(
    *,
    customer_id: str | None,
    merchant: str | None,
    country: str | None,
    predicted_label: str | None,
) -> TransactionListFilters:
    """Build read-model filters from raw query strings (raises ``HTTPException`` when invalid)."""
    return TransactionListFilters(
        customer_id=optional_trimmed(customer_id, max_length=128, field="customer_id"),
        merchant=optional_trimmed(merchant, max_length=255, field="merchant"),
        country=optional_trimmed(country, max_length=8, field="country"),
        predicted_label=optional_predicted_label(predicted_label),
    )
