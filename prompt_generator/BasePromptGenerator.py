"""Base prompt template abstraction shared by model-family prompt generators."""

from abc import ABC, abstractmethod
from typing import Optional

from enums.ModelTypeEnum import ModelTypeEnum
from utils.StringUtils import StringUtils


class BasePromptGenerator(ABC):
    """Build model-family prompts from system and user text."""

    _model_type: Optional[ModelTypeEnum] = None

    def __init__(self, model_type: ModelTypeEnum):
        """Initialize the instance and validate its configuration."""
        self.model_type = model_type

    @property
    def model_type(self) -> ModelTypeEnum:
        """Model type."""
        if self._model_type is None:
            raise RuntimeError("model_type has not been initialized.")
        return self._model_type

    @model_type.setter
    def model_type(self, model_type: ModelTypeEnum):
        """Set the model_type value after validation."""
        if model_type is None or not isinstance(model_type, ModelTypeEnum):
            raise ValueError("Model type must be not None and an instance of ModelTypeEnum.")
        self._model_type = model_type

    @staticmethod
    @abstractmethod
    def _template_for_base_models(system: str, user: str) -> str:
        """Render the prompt template for base models."""
        pass

    @staticmethod
    @abstractmethod
    def _template_for_instruct_models(who_you_are: str, system: str, user: str) -> str:
        """Render the prompt template for instruction-tuned models."""
        pass

    def generate_prompt(self, system: str, user: str, who_you_are: Optional[str] = None) -> str:
        """Generate a model-specific prompt from system and user text."""
        def clean_text(text: str) -> str:
            """Clean text."""
            text = StringUtils.trim_string(text)
            return StringUtils.remove_spaces(text)

        system = clean_text(system)
        user = clean_text(user)

        if self.model_type == ModelTypeEnum.INSTRUCT:
            if who_you_are is None:
                raise ValueError("'who_you_are' cannot be None for INSTRUCT tasks.")
            who_you_are = clean_text(who_you_are)
            return self._template_for_instruct_models(who_you_are, system, user)
        elif self.model_type == ModelTypeEnum.BASE:
            return self._template_for_base_models(system, user)
        else:
            raise ValueError(f"Unsupported ModelTypeEnum: {self.model_type}.")
