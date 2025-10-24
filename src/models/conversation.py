from pydantic import BaseModel
from typing import List


class ConversationTurn(BaseModel):
    """Individual turn in a conversation."""

    role: str
    content: str


class Conversation(BaseModel):
    """Complete conversation model."""

    turns: List[ConversationTurn]
    domain: str
    action: str
    description: str
