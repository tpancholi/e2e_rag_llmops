import sys

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from multi_doc_chat.config.settings import settings
from multi_doc_chat.exception.custom_exception import DocumentPortalExceptionError
from multi_doc_chat.logger import GLOBAL_LOGGER
from multi_doc_chat.utils.config_loader import load_config


class ModelLoader:
    """
    Loads embedding model and LLMs based on config and environment variables.
    Supports multiple vendors: OpenAI, Google, Groq.
    """

    def __init__(self, embedding_vendor: str | None = None, llm_vendor: str | None = None) -> None:
        """
        Initialize ModelLoader with vendor selection.

        Args:
            embedding_vendor: Vendor for embeddings (openai, google)
            llm_vendor: Vendor for LLM (openai, google, groq)
        """
        self.config = load_config()
        # Use vendors from settings if not provided
        self.embedding_vendor = (embedding_vendor or settings.embedding_vendor).lower()
        self.llm_vendor = (llm_vendor or settings.llm_vendor).lower()
        # Validate settings before proceeding
        try:
            settings.validate_required_keys()
        except ValueError as e:
            GLOBAL_LOGGER.error("Settings validation failed", error=str(e))
            raise

        GLOBAL_LOGGER.info(
            "Initialized ModelLoader",
            env=settings.env,
            embedding_vendor=self.embedding_vendor,
            llm_vendor=self.llm_vendor,
            config_keys=list(self.config.keys()),
        )

    def _load_openai_embeddings(self, model_name: str) -> OpenAIEmbeddings:
        """Load OpenAI embeddings model."""
        api_key = settings.get_api_key("openai")
        if not api_key:
            self._raise_missing_key_error("OPENAI_API_KEY")
        return OpenAIEmbeddings(model=model_name, api_key=api_key)

    def _load_google_embeddings(self, model_name: str) -> GoogleGenerativeAIEmbeddings:
        """Load Google embeddings model."""
        api_key = settings.get_api_key("google")
        if not api_key:
            self._raise_missing_key_error("GOOGLE_API_KEY")
        return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)

    @staticmethod
    def _raise_vendor_error(model_type: str, vendor: str, available: list[str]) -> None:
        """Raise ValueError for a vendor not found in config."""
        msg = f"{model_type} vendor '{vendor}' not found in config"
        GLOBAL_LOGGER.error(msg, available_vendors=available)
        raise ValueError(msg)

    @staticmethod
    def _raise_unsupported_provider_error(model_type: str, provider: str) -> None:
        """Raise ValueError for unsupported provider."""
        msg = f"Unsupported {model_type} provider: {provider}"
        GLOBAL_LOGGER.error(msg)
        raise ValueError(msg)

    @staticmethod
    def _raise_missing_key_error(key: str) -> None:
        """Raise ValueError for the missing API key."""
        msg = f"Missing required API key {key}"
        GLOBAL_LOGGER.error(msg, missing_key=key)
        raise ValueError(msg)

    def load_embeddings(self):
        """Load embeddings model based on config."""

        try:
            embedding_config = self.config.get("embedding_model", {})
            if self.embedding_vendor not in embedding_config:
                self._raise_vendor_error("Embedding", self.embedding_vendor, list(embedding_config.keys()))
            vendor_config = embedding_config[self.embedding_vendor]
            provider = vendor_config.get("provider")
            model_name = vendor_config.get("model_name")
            GLOBAL_LOGGER.info("Loading embedding model", provider=provider, model_name=model_name)
            if provider == "openai":
                return self._load_openai_embeddings(model_name)
            if provider == "google":
                return self._load_google_embeddings(model_name)
            self._raise_unsupported_provider_error("embedding", provider)
        except Exception as e:
            msg = "Failed to load embedding model"
            GLOBAL_LOGGER.error(msg, error=str(e))
            raise DocumentPortalExceptionError(msg, sys) from e

    def _load_openai_llm(self, model_name: str, temperature: float, max_tokens: int) -> ChatOpenAI:
        """Load OpenAI LLM."""
        api_key = settings.get_api_key("openai")
        if not api_key:
            self._raise_missing_key_error("OPENAI_API_KEY")

        # GPT-5 models don't support temperature parameter (hypothetical future models)
        if model_name.startswith("gpt-5"):
            return ChatOpenAI(model=model_name, max_tokens=max_tokens, api_key=api_key)
        return ChatOpenAI(model=model_name, max_tokens=max_tokens, temperature=temperature, api_key=api_key)

    def _load_google_llm(self, model_name: str, temperature: float, max_tokens: int) -> ChatGoogleGenerativeAI:
        """Load Google Generative AI LLM."""
        api_key = settings.get_api_key("google")
        if not api_key:
            self._raise_missing_key_error("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=model_name, temperature=temperature, max_output_tokens=max_tokens, google_api_key=api_key
        )

    def _load_groq_llm(self, model_name: str, temperature: float, max_tokens: int) -> ChatGroq:
        """Load Groq LLM."""
        api_key = settings.get_api_key("groq")
        if not api_key:
            self._raise_missing_key_error("GROQ_API_KEY")
        return ChatGroq(model=model_name, temperature=temperature, max_tokens=max_tokens, api_key=api_key)

    def load_llms(self):
        """Load LLMs based on config and selected vendor."""

        try:
            llm_config = self.config.get("llm_model", {})
            if self.llm_vendor not in llm_config:
                self._raise_vendor_error("LLM", self.llm_vendor, list(llm_config.keys()))
            vendor_config = llm_config[self.llm_vendor]
            provider = vendor_config.get("provider")
            model_name = vendor_config.get("model_name")
            temperature = vendor_config.get("temperature", 0.0)
            max_tokens = vendor_config.get("max_tokens")
            if max_tokens is None:
                max_tokens = vendor_config.get("max_output_tokens", 1024)
            GLOBAL_LOGGER.info("Loading LLM model", provider=provider, model_name=model_name)

            if provider == "openai":
                return self._load_openai_llm(model_name, temperature, max_tokens)
            if provider == "google":
                return self._load_google_llm(model_name, temperature, max_tokens)
            if provider == "groq":
                return self._load_groq_llm(model_name, temperature, max_tokens)
            self._raise_unsupported_provider_error("LLM", provider)
        except Exception as e:
            msg = "Failed to load LLM model"
            GLOBAL_LOGGER.error(msg, error=str(e))
            raise DocumentPortalExceptionError(msg, sys) from e


if __name__ == "__main__":
    # Load with default vendors (OpenAI for both)
    loader = ModelLoader()
    embeddings = loader.load_embeddings()
    llm = loader.load_llms()
    result = embeddings.embed_query("Hello world!")
    print(f"Embedding result: {result}")  # noqa: T201
    response = llm.invoke("Hello, How are you?")
    print(f"LLM response: {response}")  # noqa: T201

    # Load with specific vendors
    loader = ModelLoader(embedding_vendor="google", llm_vendor="groq")
    embeddings = loader.load_embeddings()
    llm = loader.load_llms()
    result = embeddings.embed_query("Hello world!")
    print(f"Embedding result: {result}")  # noqa: T201
    response = llm.invoke("Hello, How are you?")
    print(f"LLM response: {response}")  # noqa: T201

    # Load with different combinations
    loader = ModelLoader(embedding_vendor="openai", llm_vendor="google")
    embeddings = loader.load_embeddings()
    llm = loader.load_llms()
    result = embeddings.embed_query("Hello world!")
    print(f"Embedding result: {result}")  # noqa: T201
    response = llm.invoke("Hello, How are you?")
    print(f"LLM response: {response}")  # noqa: T201
