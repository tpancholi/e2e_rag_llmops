from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from multi_doc_chat.exception.custom_exception import DocumentPortalExceptionError
from multi_doc_chat.logger import GLOBAL_LOGGER

if TYPE_CHECKING:
    from collections.abc import Iterable

SUPPORTED_FILE_EXTENSIONS = [".pdf", ".docx", ".txt"]


def save_uploaded_file(uploaded_files: Iterable[Any], destination: Path) -> list[Path]:
    """
    Save uploaded files to the specified destination directory.

    Args:
        uploaded_files: Iterable of file objects with read() or getbuffer() methods
        destination: Path to save directory

    Returns:
        List of paths to saved files

    Raises:
        DocumentPortalExceptionError: If file saving fails
    """
    try:
        destination.mkdir(parents=True, exist_ok=True)
        saved_files: list[Path] = []

        for file in uploaded_files:
            try:
                saved_file_path = _save_single_file(file, destination)
                if saved_file_path:
                    saved_files.append(saved_file_path)
            except Exception as e:
                file_name = getattr(file, "file_name", "unknown_file")
                GLOBAL_LOGGER.error("Failed to save individual file", file_name=file_name, error=str(e))
                continue

        return saved_files  # noqa: TRY300

    except Exception as e:
        msg = "Failed to save uploaded files"
        GLOBAL_LOGGER.error(msg, error=str(e), dir=str(destination))
        raise DocumentPortalExceptionError(msg, e) from e


def _save_single_file(file: Any, destination: Path) -> Path | None:
    """Save a single file and return its path, or None if skipped."""
    file_name = getattr(file, "file_name", "unnamed_file")
    file_extension = Path(file_name).suffix.lower()

    # Check if file extension is supported
    if file_extension not in SUPPORTED_FILE_EXTENSIONS:
        GLOBAL_LOGGER.warning("Unsupported file skipped", file_name=file_name)
        return None

    # Generate safe filename
    safe_file_name = _generate_safe_filename(file_name)
    final_file_name = f"{safe_file_name}_{uuid.uuid4().hex[:6]}{file_extension}"
    saved_file_path = destination / final_file_name

    # Save file content
    with saved_file_path.open("wb") as f:
        if hasattr(file, "read"):
            f.write(file.read())
        else:
            f.write(file.getbuffer())

    GLOBAL_LOGGER.info("File saved", uploaded=file_name, saved_as=str(saved_file_path))
    return saved_file_path


def _generate_safe_filename(file_name: str) -> str:
    """Generate a safe filename by sanitizing the stem."""
    stem = Path(file_name).stem
    # Keep alphanumeric, hyphens, underscores, dots, and spaces; replace others with underscore
    safe_stem = re.sub(r"[^\w\-_. ]", "_", stem).lower()
    # Remove multiple consecutive underscores
    safe_stem = re.sub(r"_{2,}", "_", safe_stem)
    # Strip leading/trailing underscores and spaces
    return safe_stem.strip("_ ")
