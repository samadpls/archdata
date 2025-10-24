import os
import sys
from dotenv import load_dotenv
from src.config import Config
from src.pipeline import ArchRouterPipeline


def main():
    """Main entry point for the Arch-Router dataset generation pipeline."""
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: Please set GROQ_API_KEY in your .env file")
        sys.exit(1)

    try:
        config = Config()
        pipeline = ArchRouterPipeline(config, api_key)

        print("=" * 50)
        print("Arch-Router Dataset Generation Pipeline")
        print("=" * 50)
        print(f"Configuration:")
        print(f"  Model Name: {config.model_name}")
        print(f"  Target Dataset Size: {config.target_dataset_size}")
        print(f"  Max Conversation Turns: {config.max_conversation_turns}")
        print(f"  Alignment Threshold: {config.alignment_threshold}")
        print(f"  Output File: {config.output_file}")
        print("=" * 50)

        dataset = pipeline.run_pipeline()
        pipeline.save_dataset(dataset)

        print("=" * 50)
        print("Pipeline completed successfully!")
        print(f"Generated {len(dataset)} samples")
        print("=" * 50)

    except Exception as e:
        print(f"Error: Pipeline failed - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
