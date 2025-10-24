# Usage Guide

## Quick Start

### Basic Usage

```bash
python main.py
```

### Custom Configuration

```python
from src.config import Config
from src.pipeline import ArchRouterPipeline
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Customize your pipeline
config = Config(
    target_dataset_size=50,  # Generate 50 samples
    max_conversation_turns=6,
    alignment_threshold=0.85,
    output_file="my_custom_dataset.jsonl"
)

pipeline = ArchRouterPipeline(config, api_key)
dataset = pipeline.run_pipeline()
pipeline.save_dataset(dataset)
```

## Configuration Options

### Dataset Control
- `target_dataset_size`: **Main control parameter** - determines the final dataset size
- `max_samples_per_intent`: Maximum samples per intent from source data
- `max_conversation_turns`: Maximum turns in generated conversations
- `min_conversation_turns`: Minimum turns in generated conversations

### LLM Parameters
- `model_name`: Groq model to use (default: "llama-3.1-8b-instant")
- `alignment_threshold`: Score threshold for conversation-policy alignment (0.0-1.0)
- `use_domain_mixing`: Enable domain mixing for additional negative samples

### Output Control
- `output_file`: Output filename for generated dataset
- `batch_size`: Batch size for processing

## Advanced Usage

### Running with Different Dataset Sizes

```python
# Small test run
config = Config(target_dataset_size=10)

# Medium dataset
config = Config(target_dataset_size=100)

# Large dataset
config = Config(target_dataset_size=1000)
```

### Custom Augmentation Ratios

The pipeline automatically handles augmentation ratios, but you can influence them by:

1. **Adjusting target dataset size**: Larger datasets get more diverse augmentations
2. **Modifying augmentation logic**: Edit `src/phase2/augmentation_module.py`
3. **Custom prompts**: Update prompt templates in `src/prompts/`

### Using Your Own Dataset

1. **Replace CLINC150**: Update `DataProcessor` to use your dataset
2. **Modify data format**: Ensure your data matches the expected format
3. **Update prompts**: Adjust prompts for your specific domain

## Testing Individual Components

```bash
# Test data processing
python tests/test_data_processor.py

# Test policy generation
python tests/test_llm1_policy_generator.py

# Test conversation synthesis
python tests/test_llm2_conversation_synthesizer.py

# Test alignment evaluation
python tests/test_llm3_alignment_evaluator.py

# Test augmentation
python tests/test_phase2_augmentation.py
```

## Output Analysis

### Understanding the Output

Each generated sample includes:
- `conversation`: The actual conversation turns
- `domain`: The domain category (e.g., "travel", "finance")
- `action`: The specific action (e.g., "book_flight", "check_balance")
- `description`: Policy description for the action
- `label_score`: Confidence score (0.0-1.0)
- `augmentation_type`: Type of augmentation applied

### Analyzing Results

```python
import json

# Load and analyze your dataset
with open('arch_router_dataset.jsonl', 'r') as f:
    samples = [json.loads(line) for line in f]

# Count augmentation types
augmentation_counts = {}
for sample in samples:
    aug_type = sample['augmentation_type']
    augmentation_counts[aug_type] = augmentation_counts.get(aug_type, 0) + 1

print("Augmentation distribution:", augmentation_counts)
```

## Troubleshooting

### Common Issues

1. `🔑 API Key Issues`: Ensure your Groq API key is set correctly
2. `💾 Memory Issues`: Reduce `target_dataset_size` for large runs
3. `⏱️ Rate Limiting`: Add delays between API calls if needed

