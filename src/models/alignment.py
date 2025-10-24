from pydantic import BaseModel


class AlignmentScore(BaseModel):
    """Alignment score model for LLM-3 evaluation."""

    score: float
    reasoning: str
    is_aligned: bool
