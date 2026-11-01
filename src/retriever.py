import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer, util

from cache import QueryCache


class Retriever:
    """Hybrid document retriever using BM25 and semantic search."""

    def __init__(
        self, index_dir: str = "data/processed"
    ) -> None:
        """Initialize the retriever with pre-built indices.

        Args:
            index_dir: Directory containing the
                pre-built index files.

        Raises:
            FileNotFoundError: If any required index
                file is missing.
        """
        print("start retrieval embeddings...")
        chunks_path = Path(index_dir) / "chunks.pkl"
        bm25_path = Path(index_dir) / "bm25_index.pkl"
        embeddings_path = (
            Path(index_dir) / "embeddings.pkl"
        )
        self.cache = QueryCache()

        if (
            not chunks_path.exists()
            or not bm25_path.exists()
            or not embeddings_path.exists()
        ):
            raise FileNotFoundError(
                "L'Index mal9inahch."
            )

        with open(chunks_path, "rb") as f:
            self.chunks: List[Dict[str, Any]] = (
                pickle.load(f)
            )

        with open(bm25_path, "rb") as f:
            self.bm25: Any = pickle.load(f)

        with open(embeddings_path, "rb") as f:
            self.embeddings: Any = pickle.load(f)

        print(
            "Embedding Model (all-MiniLM-L6-v2)"
            " l'Recherche..."
        )
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def bm25_search(
        self, query: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform lexical BM25 search.

        Args:
            query: The search query string.
            k: Number of top results to return.

        Returns:
            List of top-k chunk dictionaries.
        """
        tokenized_query = query.lower().split()
        top_chunks: List[Dict[str, Any]] = (
            self.bm25.get_top_n(
                tokenized_query, self.chunks, n=k
            )
        )
        return top_chunks

    def semantic_search(
        self, query: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform semantic vector search.

        Args:
            query: The search query string.
            k: Number of top results to return.

        Returns:
            List of top-k chunk dictionaries.
        """
        query_embedding = (
            self.embedding_model.encode(query)
        )
        scores = util.cos_sim(
            query_embedding, self.embeddings
        )[0]
        safe_k = min(k, len(scores))
        top_results = torch.topk(scores, k=safe_k)
        top_indices = top_results.indices.tolist()
        top_chunks: List[Dict[str, Any]] = [
            self.chunks[i] for i in top_indices
        ]
        return top_chunks

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining BM25 and semantic.

        Uses Reciprocal Rank Fusion (RRF) to merge
        results from both search methods.

        Args:
            query: The search query string.
            k: Number of top results to return.
            rrf_k: RRF smoothing constant.

        Returns:
            List of top-k chunk dictionaries.
        """
        cached_results: Optional[
            List[Dict[str, Any]]
        ] = self.cache.get(query, k, task="search")
        if cached_results is not None:
            print(
                "[CACHE HIT] L9inah wajed f 0ms:"
                f" '{query}'"
            )
            return cached_results
        print("START HYBRID SEARCH ...")
        bm25_results = self.bm25_search(query, k=5)
        semantic_results = self.semantic_search(
            query, k=5
        )
        chunk_scores: Dict[str, Dict[str, Any]] = {}
        for rank, chunk in enumerate(bm25_results):
            chunk_id = (
                f"{chunk['file_path']}"
                f"_{chunk['first_character_index']}"
            )
            score = 1.0 / (rrf_k + rank + 1)
            chunk_scores[chunk_id] = {
                "chunk": chunk,
                "score": score,
            }

        for rank, chunk in enumerate(semantic_results):
            chunk_id = (
                f"{chunk['file_path']}"
                f"_{chunk['first_character_index']}"
            )
            score = 1.0 / (rrf_k + rank + 1)

            if chunk_id in chunk_scores:
                chunk_scores[chunk_id][
                    "score"
                ] += score
            else:
                chunk_scores[chunk_id] = {
                    "chunk": chunk,
                    "score": score,
                }

        sorted_results = sorted(
            chunk_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )
        top_chunks: List[Dict[str, Any]] = [
            res["chunk"] for res in sorted_results[:k]
        ]
        return top_chunks


if __name__ == "__main__":
    retriever = Retriever()

    # Ntestiw b sou2al fih klamat mbedlin
    # (Synonyms) bash nchoufo shkon ghayfhemna
    test_query = (
        "What is the procedure to execute"
        " the application?"
    )

    print("\n" + "=" * 50)
    print(f"SOU2AL: '{test_query}'")
    print("=" * 50)

    # print(
    #     "\n🔍 1. BM25 (Lexical Search)"
    #     " kay9leb..."
    # )
    # bm25_res = retriever.bm25_search(
    #     test_query, k=1
    # )
    # print(f"File: {bm25_res[0]['file_path']}")
    # print(
    #     f"Text: {bm25_res[0]['text'][:150]}..."
    # )

    # print(
    #     "\n🧠 2. Semantic (Vector Search)"
    #     " kay9leb..."
    # )
    # sem_res = retriever.semantic_search(
    #     test_query, k=1
    # )
    # print(f"File: {sem_res[0]['file_path']}")
    # print(
    #     f"Text: {sem_res[0]['text'][:150]}..."
    # )

    print(
        "\n🤝 3. HYBRID Search"
        " (BM25 + Semantic) kay9leb..."
    )
    hybrid_res = retriever.hybrid_search(
        test_query, k=2
    )
    print(f"File: {hybrid_res[0]['file_path']}")
    print(
        f"Text: {hybrid_res[0]['text'][:150]}..."
    )
