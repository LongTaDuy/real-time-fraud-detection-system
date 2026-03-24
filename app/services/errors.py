"""Domain-level errors for fraud scoring and persistence (no HTTP types here)."""


class FraudModelInputError(ValueError):
    """Raised when the request does not satisfy the loaded model's feature contract."""


class DuplicateTransactionError(ValueError):
    """Raised when a transaction_id that must be unique already exists."""

    def __init__(self, transaction_id: str) -> None:
        self.transaction_id = transaction_id
        super().__init__(f"transaction_id already exists: {transaction_id}")


class TransactionPayloadMismatchError(ValueError):
    """Raised when transaction_id exists but persisted fields differ from the request."""

    def __init__(self, transaction_id: str) -> None:
        self.transaction_id = transaction_id
        super().__init__(
            f"transaction_id already exists with different transaction data: {transaction_id}",
        )
