import json
import random
from groq import Groq
from typing import List
from src.models.conversation import Conversation, ConversationTurn
from src.models.augmentation import AugmentedConversation
from pydantic import BaseModel, Field
from src.prompts.phase2_paraphrase import (
    get_noise_injection_prompt,
    get_irrelevant_conversation_prompt,
    get_selective_paraphrase_prompt,
    get_domain_mixing_prompt,
)
from src.utils.conversation_formatter import format_conversation


class ConversationResponse(BaseModel):
    """Pydantic model for LLM conversation responses."""

    turns: List[ConversationTurn] = Field(..., description="List of conversation turns")

    @classmethod
    def from_llm_response(cls, content: str) -> "ConversationResponse":
        """Parse LLM response and return validated ConversationResponse."""
        import re

        # Extract JSON array from response
        json_match = re.search(r"\[[\s\S]*?\]", content)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Try markdown blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end != -1:
                    json_str = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                if end != -1:
                    json_str = content[start:end].strip()
            else:
                json_str = content

        # Clean up JSON
        json_str = json_str.strip()
        json_str = re.sub(r"#.*?\n", "", json_str)  # Remove comments
        json_str = re.sub(r"//.*?\n", "", json_str)  # Remove // comments

        # Fix common JSON issues
        json_str = json_str.replace("}\n    {", "},\n    {")  # Add missing commas
        json_str = json_str.replace("}\n{", "},\n{")  # Add missing commas

        try:
            turns_data = json.loads(json_str)
            turns = [ConversationTurn(**turn) for turn in turns_data]
            return cls(turns=turns)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise ValueError(f"Failed to parse LLM response as valid conversation: {e}")


class AugmentationModule:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float = 0.8,
        max_tokens: int = 1000,
    ):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _get_label_score(self, augmentation_type: str) -> float:
        scores = {
            "original": 0.95,
            "paraphrase": 0.9,
            "noise": 0.7,
            "irrelevant": 0.1,
            "domain_mix": 0.1,
        }
        return scores.get(augmentation_type, 0.5)

    def _parse_llm_response(self, content: str) -> List[ConversationTurn]:
        """Parse LLM response using Pydantic validation."""
        try:
            response = ConversationResponse.from_llm_response(content)
            return response.turns
        except ValueError as e:
            print(f"Failed to parse LLM response: {e}")
            raise

    def selective_paraphrase(self, conversation: Conversation) -> AugmentedConversation:
        user_turn_indices = [
            i for i, turn in enumerate(conversation.turns) if turn.role == "user"
        ]
        if not user_turn_indices:
            return AugmentedConversation(
                conversation=conversation,
                augmentation_type="paraphrase",
                label_score=self._get_label_score("paraphrase"),
            )

        # Select 1-3 random user turns (or all if less than 3)
        max_turns = min(3, len(user_turn_indices))
        min_turns = 1
        num_turns = random.randint(min_turns, max_turns)
        selected_indices = random.sample(user_turn_indices, num_turns)

        conversation_text = format_conversation(conversation)
        prompt = get_selective_paraphrase_prompt(conversation_text, selected_indices)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from LLM")

            turns = self._parse_llm_response(content)

            paraphrased = Conversation(
                turns=turns,
                domain=conversation.domain,
                action=conversation.action,
                description=conversation.description,
            )

            return AugmentedConversation(
                conversation=paraphrased,
                augmentation_type="paraphrase",
                label_score=self._get_label_score("paraphrase"),
            )
        except Exception as e:
            raise RuntimeError(f"Selective paraphrase augmentation failed: {e}")

    def inject_noise(self, conversation: Conversation) -> AugmentedConversation:
        conversation_text = format_conversation(conversation)
        prompt = get_noise_injection_prompt(conversation_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from LLM")

            turns = self._parse_llm_response(content)

            noisy_conversation = Conversation(
                turns=turns,
                domain=conversation.domain,
                action=conversation.action,
                description=conversation.description,
            )

            return AugmentedConversation(
                conversation=noisy_conversation,
                augmentation_type="noise",
                label_score=self._get_label_score("noise"),
            )
        except Exception as e:
            raise RuntimeError(f"Noise injection augmentation failed: {e}")

    def create_irrelevant_conversation(
        self, conversation: Conversation
    ) -> AugmentedConversation:
        prompt = get_irrelevant_conversation_prompt(
            conversation.domain, conversation.action
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from LLM")

            turns = self._parse_llm_response(content)

            irrelevant_conversation = Conversation(
                turns=turns,
                domain="irrelevant",
                action="irrelevant_chat",
                description="Irrelevant conversation for negative training",
            )

            return AugmentedConversation(
                conversation=irrelevant_conversation,
                augmentation_type="irrelevant",
                label_score=self._get_label_score("irrelevant"),
            )
        except Exception as e:
            raise RuntimeError(f"Irrelevant conversation generation failed: {e}")

    def create_domain_mixed_conversation(
        self, conversation: Conversation, other_conversation: Conversation
    ) -> AugmentedConversation:
        prompt = get_domain_mixing_prompt(conversation, other_conversation)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from LLM")

            turns = self._parse_llm_response(content)

            mixed_conversation = Conversation(
                turns=turns,
                domain="mixed",
                action="mixed_domains",
                description="Domain-mixed conversation for negative training",
            )

            return AugmentedConversation(
                conversation=mixed_conversation,
                augmentation_type="domain_mix",
                label_score=self._get_label_score("domain_mix"),
            )
        except Exception as e:
            raise RuntimeError(f"Domain mixing augmentation failed: {e}")

    def create_conversation_variants(
        self, conversation: Conversation
    ) -> List[AugmentedConversation]:
        variants = []

        variants.append(
            AugmentedConversation(
                conversation=conversation,
                augmentation_type="original",
                label_score=self._get_label_score("original"),
            )
        )

        if random.random() < 0.35:
            try:
                paraphrased = self.selective_paraphrase(conversation)
                variants.append(paraphrased)
            except Exception as e:
                print(f"Paraphrase failed: {e}")

        if random.random() < 0.225:
            try:
                noisy = self.inject_noise(conversation)
                variants.append(noisy)
            except Exception as e:
                print(f"Noise injection failed: {e}")

        if random.random() < 0.125:
            try:
                irrelevant = self.create_irrelevant_conversation(conversation)
                variants.append(irrelevant)
            except Exception as e:
                print(f"Irrelevant conversation failed: {e}")

        return variants

    def augment_conversations(
        self, conversations: List[Conversation]
    ) -> List[AugmentedConversation]:
        all_augmented = []

        for conversation in conversations:
            variants = self.create_conversation_variants(conversation)
            all_augmented.extend(variants)

        return all_augmented

    def augment_conversations_with_mixing(
        self, conversations: List[Conversation]
    ) -> List[AugmentedConversation]:
        all_augmented = []

        domain_groups: dict[str, list[Conversation]] = {}
        for conv in conversations:
            if conv.domain not in domain_groups:
                domain_groups[conv.domain] = []
            domain_groups[conv.domain].append(conv)

        for conversation in conversations:
            variants = self.create_conversation_variants(conversation)
            all_augmented.extend(variants)

            if random.random() < 0.05 and len(domain_groups) > 1:
                other_domains = [
                    d for d in domain_groups.keys() if d != conversation.domain
                ]
                if other_domains:
                    other_domain = random.choice(other_domains)
                    other_conversation = random.choice(domain_groups[other_domain])
                    try:
                        mixed = self.create_domain_mixed_conversation(
                            conversation, other_conversation
                        )
                        all_augmented.append(mixed)
                    except Exception as e:
                        print(f"Domain mixing failed: {e}")

        return all_augmented
