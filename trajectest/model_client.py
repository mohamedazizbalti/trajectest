import os
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

SUPPORTED_APIS = ("openai", "anthropic")


@dataclass
class ModelConfig:
    model: str
    api: str = "openai"
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0

    @classmethod
    def from_env(cls, prefix: str, default_model: str) -> "ModelConfig":
        return cls(
            model=os.environ.get(f"{prefix}_MODEL", default_model),
            api=os.environ.get(f"{prefix}_API", "openai"),
            base_url=os.environ.get(f"{prefix}_BASE_URL"),
            api_key_env=os.environ.get(f"{prefix}_API_KEY_ENV"),
        )


def create_chat_model(config: ModelConfig) -> BaseChatModel:
    if config.api not in SUPPORTED_APIS:
        raise ValueError(f"Unknown api '{config.api}'. Use one of: {', '.join(SUPPORTED_APIS)}")

    settings = {"model": config.model, "temperature": config.temperature}
    if config.base_url:
        settings["base_url"] = config.base_url
    if config.api_key_env:
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {config.api_key_env} is not set")
        settings["api_key"] = api_key

    if config.api == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as error:
            raise ImportError("Anthropic support needs: pip install 'trajectest[anthropic]'") from error
        return ChatAnthropic(**settings)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(**settings)
