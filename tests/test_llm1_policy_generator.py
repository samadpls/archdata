#!/usr/bin/env python3

import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.phase1.data_processor import DataProcessor
from src.phase1.llm1_policy_generator import LLM1PolicyGenerator


def test_llm1_policy_generator():
    print("=" * 50)
    print("Testing LLM-1 Policy Generator")
    print("=" * 50)

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("[ERROR] Error: Please set GROQ_API_KEY in your .env file")
        return

    config = Config()
    data_processor = DataProcessor(
        data_file="data/clinc150_uci/data_small.json", config=config
    )

    print("1. Testing data processing for LLM-1...")
    try:
        intents = data_processor.process_intents()
        print(f"[SUCCESS] Processed {len(intents)} intents with examples")

        for i, intent in enumerate(intents[:3], 1):
            print(f"  {i}. {intent.intent_name}: {len(intent.examples)} examples")
            print(
                f"     Sample: {intent.examples[0] if intent.examples else 'No examples'}"
            )
    except Exception as e:
        print(f"[ERROR] Data processing failed: {e}")
        return

    print("\n2. Testing LLM-1 Policy Generator class...")
    try:
        llm1 = LLM1PolicyGenerator(
            api_key=api_key,
            model_name=config.model_name,
            temperature=config.policy_generation_temperature,
            max_tokens=config.policy_generation_max_tokens,
        )
        print("[SUCCESS] LLM-1 Policy Generator initialized")
    except Exception as e:
        print(f"[ERROR] LLM-1 initialization failed: {e}")
        return

    print("\n3. Testing policy generation...")
    test_intents = intents[:2]

    for intent in test_intents:
        print(f"\nGenerating policy for: {intent.intent_name}")
        print(f"Examples: {intent.examples[:2]}")
        try:
            policy = llm1.generate_policy(intent.intent_name, intent.examples)
            print("[SUCCESS] Generated policy:")
            print(f"   Domain: {policy.domain}")
            print(f"   Action: {policy.action}")
            print(f"   Description: {policy.description[:100]}...")
        except Exception as e:
            print(f"[ERROR] Policy generation failed: {e}")

    print("\n4. Testing batch policy generation...")
    try:
        intents_data = [
            {"intent_name": intent.intent_name, "examples": intent.examples}
            for intent in test_intents
        ]
        policies = llm1.generate_policies_batch(intents_data)
        print(f"[SUCCESS] Generated {len(policies)} policies in batch")

        for i, policy in enumerate(policies, 1):
            print(f"  {i}. {policy.domain} -> {policy.action}")
    except Exception as e:
        print(f"[ERROR] Batch generation failed: {e}")
        return

    print("\n[SUCCESS] LLM-1 Policy Generator test completed!")
    print(f"[SUCCESS] {len(intents)} intents processed")
    print(f"[SUCCESS] {len(policies)} policies generated")


if __name__ == "__main__":
    test_llm1_policy_generator()
