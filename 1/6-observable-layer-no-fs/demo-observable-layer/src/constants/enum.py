from enum import Enum


class LLMModel(Enum):
    """
    Enum for LLM models.
    """

    OPENAI_TEXT_EMBEDDING_V4 = "text-embedding-v4"
    QWEN_3_7_FLASH = "qwen3.7-flash"


class LLMProvider(Enum):
    """
    Enum for LLM providers.
    """

    OPENAI = "openai"
