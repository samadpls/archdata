from pydantic import BaseModel


class Policy(BaseModel):
    """Policy model for LLM-1 generated policies."""

    domain: str
    action: str
    description: str
