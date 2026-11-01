"""Chunking utilities for splitting text by format."""

from typing import Any, Dict, List

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)


class LangChainChunker:
    """Splits text into chunks using LangChain splitters."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initialise the chunker.

        Args:
            max_chunk_size: Maximum characters per chunk.
        """
        self.max_chunk_size = max_chunk_size

    def chunk_markdown(
        self, text: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """Split markdown text into indexed chunks.

        Args:
            text: Raw markdown content.
            file_path: Source file path stored in each chunk.

        Returns:
            List of chunk dictionaries with position info.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=0,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""],
        )

        docs = splitter.create_documents([text])
        chunks_info: List[Dict[str, Any]] = []
        for doc in docs:
            chunk_text = doc.page_content
            start_index: int = doc.metadata["start_index"]
            end_index = start_index + len(chunk_text)

            chunks_info.append({
                "file_path": file_path,
                "first_character_index": start_index,
                "last_character_index": end_index,
                "text": chunk_text,
            })
        return chunks_info

    def chunk_python(
        self, code: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """Split Python source code into indexed chunks.

        Args:
            code: Raw Python source code.
            file_path: Source file path stored in each chunk.

        Returns:
            List of chunk dictionaries with position info.
        """
        splitter = (
            RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=self.max_chunk_size,
                chunk_overlap=0,
                add_start_index=True,
            )
        )

        docs = splitter.create_documents([code])

        chunks_info: List[Dict[str, Any]] = []
        for doc in docs:
            chunk_text = doc.page_content
            start_index: int = doc.metadata["start_index"]
            end_index = start_index + len(chunk_text)

            chunks_info.append({
                "file_path": file_path,
                "first_character_index": start_index,
                "last_character_index": end_index,
                "text": chunk_text,
            })
        return chunks_info
