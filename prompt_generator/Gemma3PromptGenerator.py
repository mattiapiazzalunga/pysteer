"""Gemma 3 prompt template implementation."""

from prompt_generator.BasePromptGenerator import BasePromptGenerator


class Gemma3PromptGenerator(BasePromptGenerator):
    """Generate Gemma 3 compatible prompts."""

    @staticmethod
    def _template_for_base_models(system: str, user: str) -> str:
        """Render the prompt template for base models."""
        return f"<bos>{system}\n\n{user}\n"

    @staticmethod
    def _template_for_instruct_models(who_you_are: str, system: str, user: str) -> str:
        """Render the prompt template for instruction-tuned models."""
        return (
            "<bos><start_of_turn>user\n"
            f"You are an expert in {who_you_are}. {system}\n\n{user}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    # The generate_prompt() method is inherited from BasePromptGenerator.
