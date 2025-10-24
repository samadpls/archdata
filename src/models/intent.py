from pydantic import BaseModel
from typing import List


class IntentData(BaseModel):
    """Intent data model from CLINC150."""

    domain: str
    action: str
    intent_name: str
    examples: List[str]
