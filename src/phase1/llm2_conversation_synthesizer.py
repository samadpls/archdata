import json
import random
from groq import Groq
from typing import List
from src.models.policy import Policy
from src.models.conversation import Conversation, ConversationTurn
from src.prompts.llm2_conversation_synthesizer import get_conversation_generation_prompt


class LLM2ConversationSynthesizer:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float = 0.8,
        min_turns: int = 2,
        max_turns: int = 5,
        max_tokens: int = 1000,
    ):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.max_tokens = max_tokens

    def generate_conversation(self, policy: Policy) -> Conversation:
        """Generate a conversation following the given policy."""
        num_turns = random.randint(self.min_turns, self.max_turns)
        prompt = get_conversation_generation_prompt(
            policy.description, policy.domain, policy.action, num_turns
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON from response (handle cases where LLM adds extra text)
            start_idx = content.find("[")
            end_idx = content.rfind("]") + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON array found in LLM response")

            json_content = content[start_idx:end_idx]

            # Try to parse the JSON, if it fails, try to find the first complete JSON array
            try:
                turns_data = json.loads(json_content)
            except json.JSONDecodeError:
                # Find the first complete JSON array
                lines = content.split("\n")
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith("["):
                        in_json = True
                        json_lines.append(line)
                    elif in_json:
                        json_lines.append(line)
                        if line.strip().endswith("]"):
                            break

                json_content = "\n".join(json_lines)
                turns_data = json.loads(json_content)
            turns = [ConversationTurn(**turn) for turn in turns_data]

            return Conversation(
                turns=turns,
                domain=policy.domain,
                action=policy.action,
                description=policy.description,
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
        except Exception as e:
            raise RuntimeError(f"LLM-2 conversation generation failed: {e}")

    def generate_conversations_batch(
        self, policies: List[Policy]
    ) -> List[Conversation]:
        """Generate conversations for a batch of policies."""
        conversations = []
        for policy in policies:
            conversation = self.generate_conversation(policy)
            conversations.append(conversation)
        return conversations
