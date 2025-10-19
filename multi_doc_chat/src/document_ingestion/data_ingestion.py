from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter

from multi_doc_chat.exception.custom_exception import DocumentPortalExceptionError
from multi_doc_chat.logger import GLOBAL_LOGGER
from multi_doc_chat.utils.model_loader import ModelLoader

if TYPE_CHECKING:
    from langchain.schema import Document


class DocumentIngestion:
    def __init__(
        self,
        data_base: str = "data",
        faiss_base: str = "vector_store",
        use_session_dirs: bool = True,
        session_id: str | None = None,
    ):
        """
        Initializes a document ingestion system with specified database and FAISS base directories.

        Args:
            data_base: Path to the base directory for storing data.
            faiss_base: Path to the base directory for storing FAISS files.
            use_session_dirs: Whether to use session-specific directories.
            session_id: Optional session identifier; generates unique ID if not provided.

        Raises:
            DocumentPortalExceptionError: If initialization fails.
        """
        try:
            self.model_loader = ModelLoader(embedding_vendor="openai", llm_vendor="openai")
            self.use_session_dirs = use_session_dirs
            self.session_id = session_id or self._generate_session_id()

            self.data_base = Path(data_base)
            self.faiss_base = Path(faiss_base)

            # Validate paths
            self._validate_paths()

            self.data_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base.mkdir(parents=True, exist_ok=True)

            self.data_dir = self._resolve_dir(self.data_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            GLOBAL_LOGGER.info(
                "Document ingestion initialized",
                session_id=self.session_id,
                data_base=str(self.data_base),
                faiss_base=str(self.faiss_base),
                data_dir=str(self.data_dir),
                faiss_dir=str(self.faiss_dir),
            )
        except Exception as e:
            msg = "Failed to initialize DocumentIngestion"
            GLOBAL_LOGGER.error(msg, error=str(e))
            raise DocumentPortalExceptionError(msg) from e

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a unique session ID with timestamp."""
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"session_{timestamp}_{unique_id}"

    def _validate_paths(self) -> None:
        """Validate that the base paths are valid."""
        if not self.data_base or not str(self.data_base).strip():
            msg = "data_base cannot be empty"
            raise ValueError(msg)
        if not self.faiss_base or not str(self.faiss_base).strip():
            msg = "faiss_base cannot be empty"
            raise ValueError(msg)

    def _resolve_dir(self, base_dir: Path) -> Path:
        """
        Resolve and return the directory path based on session settings.

        Args:
            base_dir: The base directory path.

        Returns:
            The resolved directory path.
        """
        if self.use_session_dirs:
            session_dir = base_dir / self.session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            return session_dir
        return base_dir

    @staticmethod
    def _split_document_to_chunks(
        docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 100
    ) -> list[Document]:
        """
        Split documents into smaller chunks using RecursiveCharacterTextSplitter.

        Args:
            docs: List of Document objects to split.
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of overlapping characters between chunks.

        Returns:
            List of Document objects representing the split chunks.
        """
        if not docs:
            GLOBAL_LOGGER.warning("No documents provided for splitting")
            return []

        if chunk_overlap >= chunk_size:
            GLOBAL_LOGGER.warning(
                "Chunk overlap >= chunk size, adjusting overlap", chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunk_overlap = chunk_size // 2

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            keep_separator=True,
            separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        )

        try:
            chunks = splitter.split_documents(docs)
            GLOBAL_LOGGER.info(
                "Documents split into chunks",
                original_docs=len(docs),
                num_chunks=len(chunks),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            return chunks  # noqa: TRY300
        except Exception as e:
            msg = "Failed to split documents"
            GLOBAL_LOGGER.error(msg, error=str(e))
            raise DocumentPortalExceptionError(msg) from e

    def get_session_info(self) -> dict:
        """Get information about the current session."""
        return {
            "session_id": self.session_id,
            "data_dir": str(self.data_dir),
            "faiss_dir": str(self.faiss_dir),
            "use_session_dirs": self.use_session_dirs,
        }
