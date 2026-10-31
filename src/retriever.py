import pickle
from pathlib import Path
import torch
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util

class Retriever:
    def __init__(self, index_dir: str = "data/processed"):
        print("start retrieval embeddings...")
        chunks_path = Path(index_dir) / "chunks.pkl"
        bm25_path = Path(index_dir) / "bm25_index.pkl"
        embeddings_path = Path(index_dir) / "embeddings.pkl"

        if not chunks_path.exists() or not bm25_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError("L'Index mal9inahch.")
            
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
            
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
            
        with open(embeddings_path, "rb") as f:
            self.embeddings = pickle.load(f)

        print("Embedding Model (all-MiniLM-L6-v2) l'Recherche...") 
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def bm25_search(self, query: str, k: int = 5) -> List[Dict]:
        tokenized_query = query.lower().split()
        top_chunks = self.bm25.get_top_n(tokenized_query, self.chunks, n=k)
        return top_chunks

    def semantic_search(self, query: str, k: int = 5) -> List[Dict]:
        query_embedding = self.embedding_model.encode(query)
        scores = util.cos_sim(query_embedding, self.embeddings)[0]
        safe_k = min(k, len(scores)) 
        top_results = torch.topk(scores, k=safe_k)
        top_indices = top_results.indices.tolist()
        top_chunks = [self.chunks[i] for i in top_indices]
        return top_chunks
    
    def hybrid_search(self, query: str, k: int = 5, rrf_k: int = 60) -> List[Dict]:
        bm25_results = self.bm25_search(query, k=5)
        semantic_results = self.semantic_search(query, k=5)

        chunk_scores = {}
        for rank, chunk in enumerate(bm25_results):
            chunk_id = f"{chunk['file_path']}_{chunk['first_character_index']}"
            score = 1.0 / (rrf_k + rank + 1)
            chunk_scores[chunk_id] = {"chunk": chunk, "score": score}

        for rank, chunk in enumerate(semantic_results):
            chunk_id = f"{chunk['file_path']}_{chunk['first_character_index']}"
            score = 1.0 / (rrf_k + rank + 1)

            if chunk_id in chunk_scores:
                    chunk_scores[chunk_id]["score"] += score 

            else:
                    chunk_scores[chunk_id] = {"chunk": chunk, "score": score}

        sorted_results = sorted(chunk_scores.values(), key=lambda x: x["score"], reverse=True)
        top_chunks = [res["chunk"] for res in sorted_results[:k]]
        return top_chunks



if __name__ == "__main__":
    retriever = Retriever()
    
    # Ntestiw b sou2al fih klamat mbedlin (Synonyms) bash nchoufo shkon ghayfhemna
    test_query = "What is the procedure to execute the application?"
    
    print("\n" + "="*50)
    print(f"SOU2AL: '{test_query}'")
    print("="*50)

    # print("\n🔍 1. BM25 (Lexical Search) kay9leb...")
    # bm25_res = retriever.bm25_search(test_query, k=1)
    # print(f"File: {bm25_res[0]['file_path']}")
    # print(f"Text: {bm25_res[0]['text'][:150]}...")

    # print("\n🧠 2. Semantic (Vector Search) kay9leb...")
    # sem_res = retriever.semantic_search(test_query, k=1)
    # print(f"File: {sem_res[0]['file_path']}")
    # print(f"Text: {sem_res[0]['text'][:150]}...")

    print("\n🤝 3. HYBRID Search (BM25 + Semantic) kay9leb...")
    hybrid_res = retriever.hybrid_search(test_query, k=2)
    print(f"File: {hybrid_res[0]['file_path']}")
    print(f"Text: {hybrid_res[0]['text'][:150]}...")