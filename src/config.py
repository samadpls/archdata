from pydantic import BaseModel

"""
Configuration for Arch Router Dataset Pipeline

To control the pipeline size:
- Change target_dataset_size to control the final dataset size
- This single change will affect the entire pipeline (intents, samples, etc.)
"""


class Config(BaseModel):
    # Dataset parameters
    max_samples_per_intent: int = 10
    max_conversation_turns: int = 5
    min_conversation_turns: int = 2

    # Final dataset size control
    target_dataset_size: int = 3  # testing with 3 intents

    # LLM parameters
    model_name: str = "llama-3.1-8b-instant"
    policy_generation_temperature: float = 0.7
    conversation_temperature: float = 0.8
    evaluation_temperature: float = 0.3

    # LLM token limits
    policy_generation_max_tokens: int = 200
    conversation_generation_max_tokens: int = 1000
    alignment_evaluation_max_tokens: int = 300
    augmentation_max_tokens: int = 1000

    # Alignment scoring
    alignment_threshold: float = 0.9
    max_regeneration_attempts: int = 3

    # Augmentation parameters (branching approach)
    use_domain_mixing: bool = False  # Optional domain mixing for negative samples

    # Output parameters
    output_file: str = "arch_router_dataset.jsonl"
    batch_size: int = 10
