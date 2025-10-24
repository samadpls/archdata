#!/usr/bin/env python3
"""
Test script for LLM-2 Conversation Synthesizer
Tests conversation generation from policies
"""

import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase1.llm2_conversation_synthesizer import LLM2ConversationSynthesizer
from src.models.policy import Policy


def test_llm2_conversation_synthesizer():
    """Test the LLM-2 conversation synthesizer."""
    print("=" * 50)
    print("Testing LLM-2 Conversation Synthesizer")
    print("=" * 50)

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("[ERROR] Error: Please set GROQ_API_KEY in your .env file")
        return

    synthesizer = LLM2ConversationSynthesizer(
        api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.8,
        min_turns=2,
        max_turns=4,
        max_tokens=1000,
    )

    # Test with a sample policy
    test_policy = Policy(
        domain="travel",
        action="book_flight",
        description="Assist users in booking flights by searching available options, comparing prices, and completing reservations.",
    )

    print("Testing conversation generation...")
    print(f"Policy: {test_policy.description}")

    try:
        conversation = synthesizer.generate_conversation(test_policy)
        print(
            f"\n[SUCCESS] Generated conversation with {len(conversation.turns)} turns:"
        )
        print(f"Domain: {conversation.domain}")
        print(f"Action: {conversation.action}")
        print("\nConversation:")
        for i, turn in enumerate(conversation.turns, 1):
            print(f"  {i}. {turn.role}: {turn.content}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")

    print("\n[SUCCESS] LLM-2 Conversation Synthesizer test completed!")


if __name__ == "__main__":
    test_llm2_conversation_synthesizer()
