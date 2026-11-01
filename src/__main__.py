import json
import os
from pathlib import Path

import fire
from tqdm import tqdm

from indexer import CodebaseIndexer
from model import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)
from retriever import Retriever


class RAGCLI:
    """CLI for RAG indexing, search, and dataset operations."""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Build indices from the raw codebase directory.

        Args:
            max_chunk_size: Maximum chunk size for ingestion.
        """
        raw_path = "data/raw/vllm-0.10.1"
        print(f"start Indexing dyal {raw_path} ...")
        indexer = CodebaseIndexer(
            max_chunk_size=max_chunk_size
        )
        indexer.ingest_directory(raw_path)
        indexer.token_and_build_bm25()
        indexer.build_vector_index()
        indexer.save_index()

        print(
            "Ingestion complete! Indices saved under"
            " data/processed/"
        )

    def search(self, query: str, k: int = 5) -> None:
        """Run a hybrid search for a single query.

        Args:
            query: The search query string.
            k: Number of top results to return.
        """
        retriever = Retriever()
        results = retriever.hybrid_search(query, k=k)
        print(f"\nResultats dyal: '{query}'")
        for i, res in enumerate(results):
            print(f"--- Top {i+1} ---")
            print(f"File: {res['file_path']}")
            print(
                f"Index: [{res['first_character_index']}"
                f" : {res['last_character_index']}]"
            )
            print(
                f"Text: {res['text'][:100]}...\n"
            )

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: int = 5,
    ) -> None:
        """Search over an entire dataset of questions.

        Args:
            dataset_path: Path to the JSON dataset file.
            save_directory: Directory to save results.
            k: Number of top results per question.
        """
        if not os.path.exists(dataset_path):
            print(
                f"Erreur: L'fichier {dataset_path}"
                " not fond!"
            )
            return

        with open(
            dataset_path, "r", encoding="utf-8"
        ) as f:
            dataset = json.load(f)
            questions = dataset.get(
                "rag_questions", []
            )

        retriever = Retriever()
        all_search_results = []
        minimal_sources = []
        print("Badiyin l'Search...")
        for q in tqdm(
            questions, desc="Retrieving sources"
        ):
            q_id = q.get("question_id")
            q_text = q.get("question_rg")
            retrieved_chunks = retriever.search(
                q_text, k=k
            )
            for chunk in retrieved_chunks:
                source = MinimalSource(
                    file_path=str(
                        chunk["file_path"]
                    ),
                    first_character_index=chunk[
                        "first_character_index"
                    ],
                    last_character_index=chunk[
                        "last_character_index"
                    ],
                )
                print(f"question :: {q_text}")
                print(chunk["text"])
                exit(1)
                minimal_sources.append(source)
            search_res = MinimalSearchResults(
                question_id=q_id,
                question=q_text,
                retrieved_sources=minimal_sources,
            )
            all_search_results.append(search_res)

        final_output = StudentSearchResults(
            search_results=all_search_results, k=k
        )
        Path(save_directory).mkdir(
            parents=True, exist_ok=True
        )
        file_name = Path(dataset_path).name
        save_path = Path(save_directory) / file_name

        with open(
            save_path, "w", encoding="utf-8"
        ) as f:
            f.write(
                final_output.model_dump_json(indent=4)
            )
        print(
            "\n\u2705 Saved student_search_results"
            f" to {save_path}"
        )


if __name__ == "__main__":
    fire.Fire(RAGCLI)
