from src.utils.conversation_formatter import format_conversation
from typing import List


def get_noise_injection_prompt(conversation_text: str) -> str:
    """Generate prompt for noise injection into conversation."""
    return f"""You are a conversation noise injection expert. Your task is to add realistic interruptions and distractions to the given conversation.

IMPORTANT RULES:
1. Add 1-2 realistic noise interruptions (phone calls, distractions, etc.)
2. Keep the original conversation flow intact
3. Make noise turns sound natural and believable
4. Return ONLY a valid JSON array, no other text

Original conversation:
{conversation_text}

Return the conversation with noise injected as a JSON array:
[
    {{"role": "user", "content": "original or noisy message"}},
    {{"role": "assistant", "content": "original or noisy response"}},
    ...
]

JSON Response:"""


def get_irrelevant_conversation_prompt(domain: str, action: str) -> str:
    """Generate prompt for creating irrelevant conversations."""
    return f"""You are a conversation generation expert. Your task is to create a completely irrelevant conversation that has nothing to do with the given domain and action.

IMPORTANT RULES:
1. Create a conversation about a completely different topic (cooking, sports, weather, etc.)
2. Make it natural and engaging (4-8 turns)
3. Avoid any reference to the given domain or action
4. Return ONLY a valid JSON array, no other text

Domain to avoid: {domain}
Action to avoid: {action}

Return an irrelevant conversation as a JSON array:
[
    {{"role": "user", "content": "irrelevant user message"}},
    {{"role": "assistant", "content": "irrelevant assistant response"}},
    ...
]

JSON Response:"""


def get_selective_paraphrase_prompt(
    conversation_text: str, selected_indices: List[int]
) -> str:
    """Generate prompt for selective paraphrasing of specific user turns."""
    return f"""You are a conversation paraphrasing expert. Your task is to paraphrase ONLY the user turns at positions {selected_indices} in the given conversation.

IMPORTANT RULES:
1. Only paraphrase the user turns at the specified positions
2. Keep ALL assistant responses exactly the same
3. Maintain the same conversation flow and meaning
4. Return ONLY a valid JSON array, no other text

Original conversation:
{conversation_text}

Return the conversation as a JSON array with only the specified user turns paraphrased:
[
    {{"role": "user", "content": "paraphrased or original user message"}},
    {{"role": "assistant", "content": "original assistant response"}},
    ...
]

JSON Response:"""


def get_domain_mixing_prompt(conversation1, conversation2) -> str:
    """Generate prompt for creating domain-mixed conversations."""
    conv1_text = format_conversation(conversation1)
    conv2_text = format_conversation(conversation2)

    return f"""
Create a mixed conversation by splicing these two conversations from different domains.
The result should be confusing and not belong to either domain clearly.
Mix turns from both conversations to create a negative training sample.

Conversation 1 ({conversation1.domain}):
{conv1_text}

Conversation 2 ({conversation2.domain}):
{conv2_text}

Return a mixed conversation as a JSON array of turns:
[
    {{"role": "user", "content": "mixed user message"}},
    {{"role": "assistant", "content": "mixed assistant response"}},
    ...
]

Example output:
[
    {{"role": "user", "content": "I need to book a flight to Paris"}},
    {{"role": "assistant", "content": "I can help with that. When would you like to travel?"}},
    {{"role": "user", "content": "Actually, what's my account balance?"}},
    {{"role": "assistant", "content": "I'm sorry, I can't access your account information. I'm here to help with flight bookings."}}
]
"""
