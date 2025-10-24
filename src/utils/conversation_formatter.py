from src.models.conversation import Conversation


def format_conversation(conversation: Conversation) -> str:
    """Format conversation for display."""
    formatted = []
    for turn in conversation.turns:
        formatted.append(f"{turn.role}: {turn.content}")
    return "\n".join(formatted)
