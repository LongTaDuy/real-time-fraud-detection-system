from pydantic import BaseModel, ConfigDict, Field


class PaginationMeta(BaseModel):
    """Standard offset pagination metadata."""

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
    total: int = Field(ge=0)
