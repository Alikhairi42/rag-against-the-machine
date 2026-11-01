"""Codebase indexer with BM25 and vector embeddings."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from chunker import LangChainChunker
from incremental import IncrementalTracker
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class CodebaseIndexer:
    """Index a codebase using BM25 and dense vector embeddings."""

    def __init__(
        self, max_chunk_size: int = 2000
    ) -> None:
        """Initialise the indexer.

        Args:
            max_chunk_size: Maximum size of each chunk.
        """
        self.chunker = LangChainChunker(
            max_chunk_size=max_chunk_size
        )
        self.chunks: List[Dict[str, Any]] = []
        print(
            "Kan-loadiw Embedding Model "
            "(all-MiniLM-L6-v2)..."
        )
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )
        self.embeddings: Optional[Any] = None
        self.tracker = IncrementalTracker()

    def ingest_directory(
        self, raw_data_path: str
    ) -> None:
        """Read and chunk files from *raw_data_path*.

        Args:
            raw_data_path: Root directory to scan.
        """
        print(
            f"read data in this path : {raw_data_path}"
        )

        base_path = Path(raw_data_path)
        modified_files_count = 0

        chunks_path = Path("data/processed/chunks.pkl")
        all_chunks: Dict[str, List[Dict[str, Any]]] = {}
        if chunks_path.exists():
            try:
                with open(chunks_path, "rb") as f:
                    old_chunks = pickle.load(f)
                    for c in old_chunks:
                        fp = str(c['file_path'])
                        if fp not in all_chunks:
                            all_chunks[fp] = []
                        all_chunks[fp].append(c)
            except Exception:
                all_chunks = {}

        lst = base_path.rglob("*")

        for file_path in lst:
            if file_path.name.startswith("."):
                continue
            ex = file_path.suffix.lower()

            if ex in [".md", ".py", ".txt"]:
                str_path = str(file_path)

                if self.tracker.is_file_modified(
                    str_path
                ):
                    modified_files_count += 1
                    try:
                        with open(
                            file_path,
                            "r",
                            encoding="utf-8",
                        ) as f:
                            text = f.read()

                        if ex in [".md", ".txt"]:
                            new_chunks = (
                                self.chunker
                                .chunk_markdown(
                                    text, str_path
                                )
                            )
                        else:
                            new_chunks = (
                                self.chunker
                                .chunk_python(
                                    text, str_path
                                )
                            )

                        all_chunks[str_path] = (
                            new_chunks
                        )

                    except Exception:
                        pass

        self.chunks = []
        for file_chunks in all_chunks.values():
            self.chunks.extend(file_chunks)

        print(
            f"Incremental Check: L9ina "
            f"{modified_files_count} fichier(s) "
            f"mbedlin."
        )
        print(
            f"Total chunks daba: {len(self.chunks)}"
        )

    def token_and_build_bm25(self) -> None:
        """Tokenise chunks and build a BM25 index."""
        print("start tokenized ...")

        tokenized_docs: List[List[str]] = []

        for chunk in self.chunks:
            token = chunk['text'].lower().split()
            tokenized_docs.append(token)

        print("index BM 25 ....")
        self.bm25 = BM25Okapi(tokenized_docs)

    def save_index(
        self, save_dir: str = "data/processed"
    ) -> None:
        """Persist chunks, BM25, and embeddings.

        Args:
            save_dir: Directory to write index files.
        """
        print("save index ....")
        Path(save_dir).mkdir(
            parents=True, exist_ok=True
        )

        chunks_path = Path(save_dir)
        with open(
            f"{chunks_path}/chunks.pkl", "wb"
        ) as f:
            pickle.dump(self.chunks, f)

        bm25_path = Path(save_dir)
        with open(
            f"{bm25_path}/bm25_index.pkl", "wb"
        ) as f:
            pickle.dump(self.bm25, f)

        embeddings_path = Path(save_dir)
        with open(
            f"{embeddings_path}/embeddings.pkl", "wb"
        ) as f:
            pickle.dump(self.embeddings, f)

        self.tracker.save_manifest()
        print("save index vector")

    def build_vector_index(self) -> None:
        """Encode chunk texts into dense vectors."""
        print(" Vectors (Embeddings)...")
        for chunk in self.chunks:
            texts = chunk['text']
        self.embeddings = (
            self.embedding_model.encode(
                texts, show_progress_bar=True
            )
        )
        print("vector index wajed!")


if __name__ == "__main__":
    index = CodebaseIndexer(max_chunk_size=2000)

    raw_path = "data/raw/vllm-0.10.1"

    index.ingest_directory(raw_path)
    index.token_and_build_bm25()
    index.save_index()
