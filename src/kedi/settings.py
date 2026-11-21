import dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Literal

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

# The adapter choices offered by the CLI (``--adapter``) and accepted by
# ``kedi.configure()``: the framework adapters (pydantic, langchain) plus the
# agent adapters (claude, codex, acp). ``a2a`` is a separate path, not a
# choice here. This Literal is the single source of truth; ADAPTER_CHOICES
# derives from it so the CLI option never duplicates the list. (A Literal
# cannot be unpacked from a tuple variable, so the derivation runs this way.)
AdapterName = Literal["pydantic", "langchain", "claude", "codex", "acp"]
ADAPTER_CHOICES: tuple[str, ...] = AdapterName.__args__


class Settings(BaseSettings):
    """
    Settings for running the kedi pattern application.
    """

    debug: bool = Field(False, description="Enable debug mode with verbose logging")
    adapter: AdapterName = Field("pydantic", description="Adapter to use for LLM calls")
    model_name: str = Field(
        "openai/gpt-6-luna", description="Default model name to use for LLM calls"
    )
    groq_api_key: str | None = Field(None, description="API key for Groq API access")

    model_config = SettingsConfigDict(case_sensitive=False, extra="allow")


settings = Settings.model_validate({})
