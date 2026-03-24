from decimal import Decimal
import ipaddress
import re
from typing import Annotated

from pydantic import AfterValidator, Field

MAX_AMOUNT = Decimal("999999999999.99")


def _normalize_optional_str(v: str | None) -> str | None:
    if v is None:
        return None
    s = v.strip()
    return s or None


def _validate_amount(v: Decimal) -> Decimal:
    if v <= 0:
        raise ValueError("amount must be greater than zero")
    if v > MAX_AMOUNT:
        raise ValueError("amount exceeds maximum allowed (14,2)")
    q = v.quantize(Decimal("0.01"))
    if q != v:
        raise ValueError("amount must have at most 2 decimal places")
    return q


def _validate_ip(v: str | None) -> str | None:
    v = _normalize_optional_str(v)
    if v is None:
        return None
    ipaddress.ip_address(v)
    return v


def _validate_country(v: str | None) -> str | None:
    v = _normalize_optional_str(v)
    if v is None:
        return None
    if len(v) > 8:
        raise ValueError("country must be at most 8 characters")
    return v


# Alphanumeric + common safe punctuation for external ids (adjust if your gateway uses UUIDs only)
_TX_ID_PATTERN = re.compile(r"^[\w.\-:@/]+$")


def _validate_transaction_id_chars(v: str) -> str:
    s = v.strip()
    if not _TX_ID_PATTERN.fullmatch(s):
        raise ValueError(
            "transaction_id contains invalid characters; "
            "allowed: letters, digits, underscore, dot, hyphen, colon, @, slash"
        )
    return s


StrictAmount = Annotated[Decimal, AfterValidator(_validate_amount)]
OptionalIP = Annotated[str | None, AfterValidator(_validate_ip)]
OptionalCountry = Annotated[str | None, AfterValidator(_validate_country)]
ExternalId = Annotated[
    str,
    Field(min_length=1, max_length=128),
    AfterValidator(_validate_transaction_id_chars),
]
