#!/usr/bin/env python3
"""
Test script for LLM-3 Alignment Evaluator
Tests conversation-policy alignment scoring
"""

import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase1.llm3_alignment_evaluator import LLM3AlignmentEvaluator
from src.models.conversation import Conversation, ConversationTurn


def test_llm3_alignment_evaluator():
    """Test the LLM-3 alignment evaluator."""
    print("=" * 50)
    print("Testing LLM-3 Alignment Evaluator")
    print("=" * 50)

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("[ERROR] Error: Please set GROQ_API_KEY in your .env file")
        return

    evaluator = LLM3AlignmentEvaluator(
        api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        threshold=0.9,
        max_tokens=300,
    )

    # Test with a sample conversation
    test_conversation = Conversation(
        turns=[
            ConversationTurn(
                role="user", content="I need to book a flight to New York"
            ),
            ConversationTurn(
                role="assistant",
                content="I can help you book a flight to New York. When would you like to travel?",
            ),
            ConversationTurn(role="user", content="Next Friday would work"),
            ConversationTurn(
                role="assistant",
                content="Great! I found several flights for Friday. What time would you prefer?",
            ),
        ],
        domain="travel",
        action="book_flight",
        description="Assist users in booking flights by searching available options, comparing prices, and completing reservations.",
    )

    print("Testing alignment evaluation...")
    print("Conversation:")
    for i, turn in enumerate(test_conversation.turns, 1):
        print(f"  {i}. {turn.role}: {turn.content}")

    try:
        score = evaluator.evaluate_alignment(test_conversation)
        print("\n[SUCCESS] Alignment evaluation:")
        print(f"   Score: {score.score}")
        print(f"   Reasoning: {score.reasoning}")
        print(f"   Is Aligned: {score.is_aligned}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")

    print("\n[SUCCESS] LLM-3 Alignment Evaluator test completed!")


if __name__ == "__main__":
    test_llm3_alignment_evaluator()
