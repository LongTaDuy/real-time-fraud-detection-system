"""Defense-in-depth helpers: body limits, rate limits, client IP extraction."""

from app.security.client_ip import get_client_ip

__all__ = ["get_client_ip"]
