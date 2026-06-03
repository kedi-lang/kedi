import typing

import dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

dotenv.load_dotenv()


class Settings(BaseSettings):
    """
    Settings for running the kedi pattern application.
    """

    debug: bool = Field(False, description="Enable debug mode with verbose logging")
    adapter: typing.Literal["dspy", "pydantic"] = Field(
        "pydantic", description="Adapter to use for LLM calls"
    )
    model_name: str = Field(
        "groq:qwen/qwen3-32b", description="Default model name to use for LLM calls"
    )
    groq_api_key: typing.Optional[str] = Field(None, description="API key for Groq API access")

    class Config:
        case_sensitive = False
        extra = "allow"
        # env_prefix = "KEDI_"


settings = Settings.model_validate({})
