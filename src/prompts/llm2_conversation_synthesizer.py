def get_conversation_generation_prompt(
    policy_description: str, domain: str, action: str, num_turns: int
) -> str:
    """Generate prompt for LLM-2 conversation synthesis."""
    return f"""
Generate a realistic conversation between a user and an AI assistant.

Policy: {policy_description}
Domain: {domain}
Action: {action}

Create a conversation with exactly {num_turns} turns where:
1. The user initiates with a request related to the policy
2. The assistant responds appropriately
3. The conversation flows naturally and follows the policy

Return the conversation as a JSON array of turns:
[
    {{"role": "user", "content": "user message"}},
    {{"role": "assistant", "content": "assistant response"}},
    ...
]

Example output:
[
    {{"role": "user", "content": "I need to book a flight to New York next week"}},
    {{"role": "assistant", "content": "I can help you book a flight to New York. What date would you prefer to travel?"}},
    {{"role": "user", "content": "Friday would work best for me"}},
    {{"role": "assistant", "content": "Great! I found several flights for Friday. Would you prefer morning or evening departure?"}}
]
"""
