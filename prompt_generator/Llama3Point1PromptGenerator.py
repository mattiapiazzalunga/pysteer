"""Llama 3.1 prompt template implementation."""

from prompt_generator.BasePromptGenerator import BasePromptGenerator


class Llama3Point1PromptGenerator(BasePromptGenerator):
    """Generate Llama 3.1 compatible prompts."""

    @staticmethod
    def _template_for_base_models(system: str, user: str) -> str:
        """Render the prompt template for base models."""
        return f"<|begin_of_text|>{system}\n\n{user}\n\n"

    @staticmethod
    def _template_for_instruct_models(who_you_are: str, system: str, user: str) -> str:
        """Render the prompt template for instruction-tuned models."""
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are an expert in {who_you_are}. "
            f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    # The generate_prompt() method is inherited from BasePromptGenerator.
