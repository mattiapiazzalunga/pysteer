"""Qwen 2.5 prompt template implementation."""

from prompt_generator.BasePromptGenerator import BasePromptGenerator


class Qwen2Point5PromptGenerator(BasePromptGenerator):
    """Generate Qwen 2.5 compatible prompts."""

    @staticmethod
    def _template_for_base_models(system: str, user: str) -> str:
        """Render the prompt template for base models."""
        return f"{system}\n{user}\n"

    @staticmethod
    def _template_for_instruct_models(who_you_are: str, system: str, user: str) -> str:
        """Render the prompt template for instruction-tuned models."""
        return (
            "<|im_start|>system\n"
            f"You are an expert in {who_you_are}. "
            f"{system}<|im_end|>\n<|im_start|>user\n"
            f"{user}<|im_end|>\n<|im_start|>assistant\n"
        )
    # The generate_prompt() method is inherited from BasePromptGenerator.
