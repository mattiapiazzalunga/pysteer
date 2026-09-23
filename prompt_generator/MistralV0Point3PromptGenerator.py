"""Mistral v0.3 prompt template implementation."""

from prompt_generator.BasePromptGenerator import BasePromptGenerator


class MistralV0Point3PromptGenerator(BasePromptGenerator):
    """Generate Mistral v0.3 compatible prompts."""

    @staticmethod
    def _template_for_base_models(system: str, user: str) -> str:
        """Render the prompt template for base models."""
        return f"<s>{system}\n\n{user}"

    @staticmethod
    def _template_for_instruct_models(who_you_are: str, system: str, user: str) -> str:
        """Render the prompt template for instruction-tuned models."""
        return (
            "<s>[INST] "
            f"You are an expert in {who_you_are}. {system}\n\n{user}[/INST]"
        )
    # The generate_prompt() method is inherited from BasePromptGenerator.
