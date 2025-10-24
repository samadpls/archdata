#!/usr/bin/env python3
"""
Test script for Phase 2 Augmentation Module
Tests paraphrasing, noise injection, and irrelevant conversation generation
"""

import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase2.augmentation_module import AugmentationModule
from src.models.conversation import Conversation, ConversationTurn


def test_phase2_augmentation():
    """Test the phase 2 augmentation module."""
    print("=" * 50)
    print("Testing Phase 2 Augmentation Module")
    print("=" * 50)

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("[ERROR] Error: Please set GROQ_API_KEY in your .env file")
        return

    augmentation = AugmentationModule(
        api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.8,
        max_tokens=1000,
    )

    # Test conversation
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

    print("Original conversation:")
    for i, turn in enumerate(test_conversation.turns, 1):
        print(f"  {i}. {turn.role}: {turn.content}")

    # Test the new branching approach
    print("\n1. Testing conversation variants (branching approach)...")
    try:
        variants = augmentation.create_conversation_variants(test_conversation)
        print(f"[SUCCESS] Created {len(variants)} variants:")

        for i, variant in enumerate(variants, 1):
            print(
                f"\n  Variant {i} ({variant.augmentation_type}, score: {variant.label_score}):"
            )
            for j, turn in enumerate(variant.conversation.turns, 1):
                print(f"    {j}. {turn.role}: {turn.content}")
    except Exception as e:
        print(f"[ERROR] Variants creation error: {e}")

    # Test individual augmentation methods
    print("\n2. Testing selective paraphrasing...")
    try:
        print("Original conversation:")
        for i, turn in enumerate(test_conversation.turns, 1):
            print(f"  {i}. {turn.role}: {turn.content}")

        paraphrased = augmentation.selective_paraphrase(test_conversation)
        print(
            f"\n[SUCCESS] Selective paraphrased conversation ({paraphrased.augmentation_type}):"
        )
        for i, turn in enumerate(paraphrased.conversation.turns, 1):
            print(f"  {i}. {turn.role}: {turn.content}")

        print("\n=== COMPARISON ===")
        print("Changes made:")
        for i, (orig, para) in enumerate(
            zip(test_conversation.turns, paraphrased.conversation.turns), 1
        ):
            if orig.content != para.content:
                print(f"  Turn {i} changed:")
                print(f"    Original: {orig.content}")
                print(f"    Paraphrased: {para.content}")
            else:
                print(f"  Turn {i}: No change")
    except Exception as e:
        print(f"[ERROR] Selective paraphrase error: {e}")

    # Test noise injection
    print("\n3. Testing noise injection...")
    try:
        print("Original conversation:")
        for i, turn in enumerate(test_conversation.turns, 1):
            print(f"  {i}. {turn.role}: {turn.content}")

        noisy = augmentation.inject_noise(test_conversation)
        print(f"\n[SUCCESS] Noisy conversation ({noisy.augmentation_type}):")
        print(f"Number of turns: {len(noisy.conversation.turns)}")
        for i, turn in enumerate(noisy.conversation.turns, 1):
            print(f"  {i}. {turn.role}: {turn.content}")

        print("\n=== NOISE ANALYSIS ===")
        print(f"Original turns: {len(test_conversation.turns)}")
        print(f"Noisy turns: {len(noisy.conversation.turns)}")
        print(
            f"Added {len(noisy.conversation.turns) - len(test_conversation.turns)} noise turns"
        )

        if len(noisy.conversation.turns) > len(test_conversation.turns):
            print("New noise turns added:")
            for i in range(len(test_conversation.turns), len(noisy.conversation.turns)):
                print(
                    f"  {i + 1}. {noisy.conversation.turns[i].role}: {noisy.conversation.turns[i].content}"
                )
    except Exception as e:
        print(f"[ERROR] Noise injection error: {e}")
        import traceback

        traceback.print_exc()

    # Test irrelevant conversation
    print("\n4. Testing irrelevant conversation...")
    try:
        print("Original conversation (travel domain):")
        for i, turn in enumerate(test_conversation.turns, 1):
            print(f"  {i}. {turn.role}: {turn.content}")

        irrelevant = augmentation.create_irrelevant_conversation(test_conversation)
        print(f"\n[SUCCESS] Irrelevant conversation ({irrelevant.augmentation_type}):")
        print(f"Number of turns: {len(irrelevant.conversation.turns)}")
        for i, turn in enumerate(irrelevant.conversation.turns, 1):
            print(f"  {i}. {turn.role}: {turn.content}")

        print("\n=== DOMAIN CHANGE ANALYSIS ===")
        print(f"Original domain: {test_conversation.domain}")
        print(f"Irrelevant domain: {irrelevant.conversation.domain}")
        print(f"Original action: {test_conversation.action}")
        print(f"Irrelevant action: {irrelevant.conversation.action}")
        print("Topic completely changed to avoid travel/booking context")
    except Exception as e:
        print(f"[ERROR] Irrelevant conversation error: {e}")
        import traceback

        traceback.print_exc()

    # Test batch augmentation
    print("\n5. Testing batch augmentation...")
    try:
        test_conversations = [test_conversation]
        augmented_batch = augmentation.augment_conversations(test_conversations)
        print(
            f"[SUCCESS] Batch augmentation created {len(augmented_batch)} total variants"
        )

        # Count by type
        type_counts = {}
        for variant in augmented_batch:
            aug_type = variant.augmentation_type
            type_counts[aug_type] = type_counts.get(aug_type, 0) + 1

        print("  Variant distribution:")
        for aug_type, count in type_counts.items():
            print(f"    {aug_type}: {count}")

    except Exception as e:
        print(f"[ERROR] Batch augmentation error: {e}")

    print("\n[SUCCESS] Phase 2 Augmentation test completed!")


if __name__ == "__main__":
    test_phase2_augmentation()
