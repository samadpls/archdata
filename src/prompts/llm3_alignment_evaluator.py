def get_alignment_evaluation_prompt(
    conversation_text: str, policy_description: str, domain: str, action: str
) -> str:
    """Generate prompt for LLM-3 alignment evaluation."""
    return f"""
Evaluate how well this conversation aligns with the given policy.

Policy: {policy_description}
Domain: {domain}
Action: {action}

Conversation:
{conversation_text}

Rate the alignment on a scale of 0.0 to 1.0 where:
- 1.0 = Perfect alignment, conversation follows policy exactly
- 0.8-0.9 = Good alignment, minor issues
- 0.6-0.7 = Moderate alignment, some problems
- 0.0-0.5 = Poor alignment, major issues

Return a JSON object:
{{"score": 0.95, "reasoning": "brief explanation", "is_aligned": true}}

Example output:
{{"score": 0.92, "reasoning": "Conversation follows the policy well, user asks for flight booking and assistant provides relevant help", "is_aligned": true}}
"""
