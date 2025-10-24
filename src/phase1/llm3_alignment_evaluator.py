import json
from groq import Groq
from typing import List
from src.models.conversation import Conversation
from src.models.alignment import AlignmentScore
from src.prompts.llm3_alignment_evaluator import get_alignment_evaluation_prompt
from src.utils.conversation_formatter import format_conversation


class LLM3AlignmentEvaluator:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float = 0.3,
        threshold: float = 0.9,
        max_tokens: int = 300,
    ):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.threshold = threshold
        self.max_tokens = max_tokens

    def evaluate_alignment(self, conversation: Conversation) -> AlignmentScore:
        """Evaluate how well a conversation aligns with its policy."""
        conversation_text = format_conversation(conversation)
        prompt = get_alignment_evaluation_prompt(
            conversation_text,
            conversation.description,
            conversation.domain,
            conversation.action,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            result = json.loads(response.choices[0].message.content.strip())
            score = float(result.get("score", 0.5))
            reasoning = result.get("reasoning", "No reasoning provided")
            is_aligned = score >= self.threshold

            return AlignmentScore(
                score=score, reasoning=reasoning, is_aligned=is_aligned
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")
        except Exception as e:
            raise RuntimeError(f"LLM-3 alignment evaluation failed: {e}")

    def evaluate_batch(self, conversations: List[Conversation]) -> List[AlignmentScore]:
        """Evaluate alignment for a batch of conversations."""
        scores = []
        for conversation in conversations:
            score = self.evaluate_alignment(conversation)
            scores.append(score)
        return scores
