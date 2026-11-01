import dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Literal

dotenv.load_dotenv()


class Settings(BaseSettings):
    """
    Settings for running the kedi pattern application.
    """

    debug: bool = Field(False, description="Enable debug mode with verbose logging")
    adapter: Literal["dspy", "pydantic"] = Field(
        "pydantic", description="Adapter to use for LLM calls"
    )
    model_name: str = Field(
        "openai/gpt-6-luna", description="Default model name to use for LLM calls"
    )
    groq_api_key: str | None = Field(None, description="API key for Groq API access")

    model_config = SettingsConfigDict(case_sensitive=False, extra="allow")


settings = Settings.model_validate({})
