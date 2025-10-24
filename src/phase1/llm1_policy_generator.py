import json
from groq import Groq
from typing import List
from src.models.policy import Policy
from src.prompts.llm1_policy_generator import get_policy_generation_prompt


class LLM1PolicyGenerator:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 200,
    ):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_policy(self, intent_name: str, examples: List[str]) -> Policy:
        """Generate a policy description for given intent name and examples."""
        prompt = get_policy_generation_prompt(intent_name, examples)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from response (handle markdown code blocks)
            if content.startswith("```json"):
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                json_content = content[start_idx:end_idx]
            elif content.startswith("```"):
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                json_content = content[start_idx:end_idx]
            else:
                json_content = content

            result = json.loads(json_content)
            return Policy(**result)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
        except Exception as e:
            raise RuntimeError(f"LLM-1 policy generation failed: {e}")

    def generate_policies_batch(self, intents_data: List[dict]) -> List[Policy]:
        """Generate policies for a batch of intent data."""
        policies = []
        for intent_data in intents_data:
            policy = self.generate_policy(
                intent_data["intent_name"], intent_data["examples"]
            )
            policies.append(policy)
        return policies
