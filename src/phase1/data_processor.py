import json
from typing import List, Dict, Tuple
from src.models.intent import IntentData


class DataProcessor:
    def __init__(self, data_file: str, config):
        self.data_file = data_file
        self.config = config
        self.min_domain_action_parts = 2
        self.domain_action_split_index = 1

    def load_clinc_data(self) -> Dict:
        """Load CLINC150 data from JSON file."""
        with open(self.data_file, "r") as f:
            return json.load(f)

    def extract_domain_action(self, intent_name: str) -> Tuple[str, str]:
        """Extract domain and action from intent name."""
        parts = intent_name.split("_")
        if len(parts) >= self.min_domain_action_parts:
            domain = parts[0]
            action = "_".join(parts[self.domain_action_split_index :])
        else:
            domain = "general"
            action = intent_name
        return domain, action

    def process_intents(self) -> List[IntentData]:
        """Process CLINC150 intents into structured data."""
        data = self.load_clinc_data()
        intents = []

        # Extract all unique intent labels from the data
        intent_labels = set()
        for split in ["train", "val", "test"]:
            if split in data:
                for item in data[split]:
                    if isinstance(item, list) and len(item) >= 2:
                        intent_labels.add(item[1])

        intent_count = 0
        for intent_name in sorted(intent_labels):
            if intent_name == "oos" or intent_count >= (
                self.config.target_dataset_size // 2
            ):
                continue

            # Collect examples for this intent from all splits
            text_examples = []
            for split in ["train", "val", "test"]:
                if split in data:
                    for item in data[split]:
                        if (
                            isinstance(item, list)
                            and len(item) >= 2
                            and item[1] == intent_name
                        ):
                            text_examples.append(item[0])
                            if len(text_examples) >= self.config.max_samples_per_intent:
                                break
                    if len(text_examples) >= self.config.max_samples_per_intent:
                        break

            intent_data = IntentData(
                domain="",  # Let LLM determine domain
                action="",  # Let LLM determine action
                intent_name=intent_name,
                examples=text_examples,
            )
            intents.append(intent_data)
            intent_count += 1

        return intents

    def get_domain_action_pairs(self) -> List[Tuple[str, str]]:
        """Get domain-action pairs from processed intents."""
        intents = self.process_intents()
        return [(intent.domain, intent.action) for intent in intents]

    def save_processed_intents(
        self, output_file: str = "processed_intents.json"
    ) -> str:
        """Save processed intent data to JSON file for LLM-1 policy generation."""
        intents = self.process_intents()

        intent_data = []
        for intent in intents:
            intent_data.append(
                {
                    "domain": intent.domain,
                    "action": intent.action,
                    "intent_name": intent.intent_name,
                    "examples": intent.examples,
                }
            )

        with open(output_file, "w") as f:
            json.dump(intent_data, f, indent=2)

        return output_file
