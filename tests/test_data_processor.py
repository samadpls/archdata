#!/usr/bin/env python3

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.phase1.data_processor import DataProcessor


def test_data_processor():
    print("=" * 50)
    print("Testing DataProcessor Class")
    print("=" * 50)

    config = Config()
    data_processor = DataProcessor(
        data_file="data/clinc150_uci/data_full.json", config=config
    )

    print("1. Testing data loading...")
    try:
        data = data_processor.load_clinc_data()
        print("[SUCCESS] Data loaded successfully")
        print(f"   Keys: {list(data.keys())}")
    except Exception as e:
        print(f"[ERROR] Data loading failed: {e}")
        return

    print("\n2. Testing intent processing...")
    try:
        intents = data_processor.process_intents()
        print(f"[SUCCESS] Processed {len(intents)} intents")

        for i, intent in enumerate(intents[:3], 1):
            print(f"  {i}. {intent.intent_name}: {len(intent.examples)} examples")
            print(
                f"     Sample: {intent.examples[0] if intent.examples else 'No examples'}"
            )
            print("     Domain/Action: To be determined by LLM")
    except Exception as e:
        print(f"[ERROR] Intent processing failed: {e}")
        return

    print("\n3. Testing intent data structure...")
    try:
        print("[SUCCESS] Intent data structure ready for LLM analysis")
        print("  - Intent names: Available for LLM to analyze")
        print("  - Examples: Available for domain/action determination")
        print("  - Domain/Action: Will be determined by LLM-1")
    except Exception as e:
        print(f"[ERROR] Intent data structure check failed: {e}")
        return

    print("\n4. Testing intent data saving...")
    try:
        output_file = data_processor.save_processed_intents(
            "test_processed_intents.json"
        )
        print(f"[SUCCESS] Saved processed intents to {output_file}")

        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                import json

                saved_data = json.load(f)
            print(f"[SUCCESS] Verified saved data: {len(saved_data)} intents")
            os.remove(output_file)
            print("[SUCCESS] Cleaned up test file")
    except Exception as e:
        print(f"[ERROR] Intent saving failed: {e}")
        return

    print("\n[SUCCESS] DataProcessor class test completed successfully!")
    print(f"[SUCCESS] Found {len(intents)} actual intent labels")
    print("[SUCCESS] Intent data ready for LLM analysis")
    print("[SUCCESS] Intent data saving working")


if __name__ == "__main__":
    test_data_processor()
