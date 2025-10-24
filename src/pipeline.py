import json
from typing import List, Dict

from src.config import Config
from src.phase1.data_processor import DataProcessor
from src.phase1.llm1_policy_generator import LLM1PolicyGenerator
from src.phase1.llm2_conversation_synthesizer import LLM2ConversationSynthesizer
from src.phase1.llm3_alignment_evaluator import LLM3AlignmentEvaluator
from src.phase2.augmentation_module import AugmentationModule


class ArchRouterPipeline:
    def __init__(self, config: Config, api_key: str):
        self.config = config
        self.api_key = api_key

        self.data_processor = DataProcessor(
            data_file="data/clinc150_uci/data_small.json", config=config
        )

        self.llm1 = LLM1PolicyGenerator(
            api_key=api_key,
            model_name=config.model_name,
            temperature=config.policy_generation_temperature,
            max_tokens=config.policy_generation_max_tokens,
        )

        self.llm2 = LLM2ConversationSynthesizer(
            api_key=api_key,
            model_name=config.model_name,
            temperature=config.conversation_temperature,
            min_turns=config.min_conversation_turns,
            max_turns=config.max_conversation_turns,
            max_tokens=config.conversation_generation_max_tokens,
        )

        self.llm3 = LLM3AlignmentEvaluator(
            api_key=api_key,
            model_name=config.model_name,
            temperature=config.evaluation_temperature,
            threshold=config.alignment_threshold,
            max_tokens=config.alignment_evaluation_max_tokens,
        )

        self.augmentation = AugmentationModule(
            api_key=api_key,
            model_name=config.model_name,
            temperature=config.conversation_temperature,
            max_tokens=config.augmentation_max_tokens,
        )

    def run_pipeline(self) -> List[Dict]:
        """Run the complete Arch-Router dataset generation pipeline."""
        print("Starting Arch-Router dataset generation pipeline...")
        print(f"Target dataset size: {self.config.target_dataset_size} samples")

        print("Step 1: Processing CLINC150 data...")
        intents = self.data_processor.process_intents()
        print(f"Processed {len(intents)} intents with examples")

        # Convert to format for LLM-1
        intents_data = []
        for intent in intents:
            intents_data.append(
                {"intent_name": intent.intent_name, "examples": intent.examples}
            )

        # Limit intents based on target dataset size
        max_intents_needed = min(
            len(intents_data), self.config.target_dataset_size // 2
        )  # Rough estimate
        intents_data = intents_data[:max_intents_needed]
        print(f"Using {len(intents_data)} intents for target size")

        print("Step 2: Generating policies with LLM-1...")
        policies = self.llm1.generate_policies_batch(intents_data)
        print(f"Generated {len(policies)} policies")

        print("Step 3: Generating conversations with LLM-2...")
        conversations = self.llm2.generate_conversations_batch(policies)
        print(f"Generated {len(conversations)} conversations")

        print("Step 4: Evaluating alignment with LLM-3...")
        alignment_scores = self.llm3.evaluate_batch(conversations)

        aligned_conversations = []
        rejected_conversations = []

        for conv, score in zip(conversations, alignment_scores):
            if score.is_aligned:
                aligned_conversations.append(conv)
            else:
                rejected_conversations.append(conv)

        print(
            f"Aligned: {len(aligned_conversations)}, Rejected: {len(rejected_conversations)}"
        )

        print("Step 5: Applying augmentations (branching approach)...")
        # Use the new branching augmentation approach
        augmented_conversations = self.augmentation.augment_conversations(
            aligned_conversations
        )

        # Optional: Use domain mixing for additional negative samples
        if hasattr(self.config, "use_domain_mixing") and self.config.use_domain_mixing:
            print("Step 5b: Adding domain mixing for negative samples...")
            mixed_conversations = self.augmentation.augment_conversations_with_mixing(
                aligned_conversations
            )
            augmented_conversations.extend(mixed_conversations)

        print("Step 6: Formatting final dataset...")

        # Show augmentation statistics
        self._show_augmentation_stats(augmented_conversations)

        final_dataset = self._format_final_dataset(augmented_conversations)

        # Limit to target dataset size
        if len(final_dataset) > self.config.target_dataset_size:
            final_dataset = final_dataset[: self.config.target_dataset_size]
            print(f"Limited dataset to target size: {len(final_dataset)} samples")

        print(
            f"Generated {len(final_dataset)} final samples (target: {self.config.target_dataset_size})"
        )
        return final_dataset

    def _format_final_dataset(self, augmented_conversations: List) -> List[Dict]:
        """Format augmented conversations into final dataset format."""
        dataset = []

        for aug_conv in augmented_conversations:
            conversation_data = {
                "conversation": [
                    {"role": turn.role, "content": turn.content}
                    for turn in aug_conv.conversation.turns
                ],
                "domain": aug_conv.conversation.domain,
                "action": aug_conv.conversation.action,
                "description": aug_conv.conversation.description,
                "label_score": aug_conv.label_score,
                "augmentation_type": aug_conv.augmentation_type,
            }
            dataset.append(conversation_data)

        return dataset

    def _show_augmentation_stats(self, augmented_conversations: List):
        """Show statistics about the augmentation results."""
        print("\n=== Augmentation Statistics ===")
        print(f"Total augmented conversations: {len(augmented_conversations)}")

        # Count by augmentation type
        type_counts = {}
        for aug_conv in augmented_conversations:
            aug_type = aug_conv.augmentation_type
            type_counts[aug_type] = type_counts.get(aug_type, 0) + 1

        print("Augmentation type distribution:")
        for aug_type, count in type_counts.items():
            percentage = (count / len(augmented_conversations)) * 100
            print(f"  {aug_type}: {count} ({percentage:.1f}%)")

        # Show label score distribution
        label_scores = [aug_conv.label_score for aug_conv in augmented_conversations]
        avg_score = sum(label_scores) / len(label_scores)
        print(f"Average label score: {avg_score:.3f}")
        print("=" * 35)

    def save_dataset(self, dataset: List[Dict], output_file: str = ""):
        """Save dataset to JSONL file."""
        if not output_file:
            output_file = self.config.output_file

        with open(output_file, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")

        print(f"Dataset saved to {output_file}")
