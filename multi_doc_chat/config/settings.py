from enum import Enum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Supported Environment names"""

    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class EmbeddingVendor(str, Enum):
    """Supported embedding vendors"""

    OPENAI = "openai"
    GOOGLE = "google"


class LLMVendor(str, Enum):
    """Supported LLM vendors"""

    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"


class AppSettings(BaseSettings):
    """Application settings to validate env variables"""

    # environment
    env: Environment = Field(default=Environment.LOCAL, description="Environment name (local, production)")

    # API Keys
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    google_api_key: str | None = Field(default=None, description="Google API key")
    groq_api_key: str | None = Field(default=None, description="Groq API key")

    # Model Config
    embedding_vendor: EmbeddingVendor = Field(
        default=EmbeddingVendor.OPENAI, description="Embedding model vendor (openai, google)"
    )
    llm_vendor: LLMVendor = Field(default=LLMVendor.OPENAI, description="LLM vendor (openai, google, groq)")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        use_enum_values=True,
    )

    def get_api_key(self, vendor: str) -> str:
        """Get API key based on specified vendor name with validation
        Args:
            vendor (str): Vendor name (openai, google, groq)
        Returns:
            API key string
        Raises:
            ValueError: If API Key is not configured
        """

        vendor_key_map = {
            "openai": ("openai_api_key", self.openai_api_key),
            "google": ("google_api_key", self.google_api_key),
            "groq": ("groq_api_key", self.groq_api_key),
        }

        if vendor not in vendor_key_map:
            msg = f"Invalid vendor name: {vendor}"
            raise ValueError(msg)

        key_name, key_value = vendor_key_map[vendor]
        if not key_value:
            msg = f"Missing required API key: {key_name.upper()}"
            raise ValueError(msg)
        return key_value

    def validate_required_keys(self) -> None:
        """Validate required API keys based on the vendor selected
        Raises:
            ValueError: If any required API key is missing
        """
        errors = []

        # check embedding a vendor key
        embedding_vendor_str = (
            self.embedding_vendor.value if isinstance(self.embedding_vendor, Enum) else self.embedding_vendor
        )
        try:
            self.get_api_key(embedding_vendor_str)
        except ValueError as e:
            errors.append(f"Embedding: {e}")

        # check llm vendor key
        llm_vendor_str = self.llm_vendor.value if isinstance(self.llm_vendor, Enum) else self.llm_vendor
        try:
            self.get_api_key(llm_vendor_str)
        except ValueError as e:
            errors.append(f"LLM: {e}")

        if errors:
            msg = "Config validation failed:\n" + "\n".join(errors)
            raise ValueError(msg)


settings = AppSettings()
