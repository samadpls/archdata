from pydantic import BaseModel
from src.models.conversation import Conversation


class AugmentedConversation(BaseModel):
    """Augmented conversation model."""

    conversation: Conversation
    augmentation_type: str
    label_score: float
