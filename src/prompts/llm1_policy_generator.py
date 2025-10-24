from typing import List


def get_policy_generation_prompt(intent_name: str, examples: List[str]) -> str:
    """Generate prompt for LLM-1 policy generation."""
    examples_text = "\n".join([f"- {example}" for example in examples])

    return f"""
Analyze the following intent and examples to determine the domain, action, and policy description.

Intent Name: {intent_name}

Examples:
{examples_text}

Based on the examples above, determine:
1. The domain (e.g., travel, banking, food, entertainment, etc.)
2. The action (what the user wants to accomplish)
3. A clear policy description

Return only a JSON object with this exact format:
{{"domain": "determined_domain", "action": "determined_action", "description": "policy description"}}

Example output:
{{"domain": "travel", "action": "book_flight", "description": "Assist users in booking flights by searching available options, comparing prices, and completing reservations."}}
"""
