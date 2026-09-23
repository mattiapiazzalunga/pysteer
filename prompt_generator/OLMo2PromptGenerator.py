"""OLMo 2 prompt template implementation."""

from prompt_generator.BasePromptGenerator import BasePromptGenerator


class OLMo2PromptGenerator(BasePromptGenerator):
    """Generate OLMo 2 compatible prompts."""

    @staticmethod
    def _template_for_base_models(system: str, user: str) -> str:
        """Render the prompt template for base models."""
        return f"{system}\n\n{user}\n"

    @staticmethod
    def _template_for_instruct_models(who_you_are: str, system: str, user: str) -> str:
        """Render the prompt template for instruction-tuned models."""
        return (
            "<|user|>\n"
            f"You are an expert in {who_you_are}. {system}\n\n{user}\n"
            "<|assistant|>\n"
        )

    # The generate_prompt() method is inherited from BasePromptGenerator.
