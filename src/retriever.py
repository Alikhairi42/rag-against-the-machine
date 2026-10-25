import pickle
from pathlib import Path
from typing import List, Dict

class BM25Retriever:
    def __init__(self, index_dir: str = "data/processed"):
        chunks_path = Path(index_dir) / "chunks.pkl"
        bm25_path = Path(index_dir) / "bm25_index.pkl"

        if not chunks_path.exists() or not bm25_path.exists():
            raise FileNotFoundError("this path not founde")
        
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
        
        print("indexs load....")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        tokenized_query = query.lower().split()

        top_chunks = self.bm25.get_top_n(tokenized_query, self.chunks, n=k)

        return top_chunks

if __name__ == "__main__":
    # Nsaybo moteur de recherche dyalna
    retriever = BM25Retriever()
    
    # Nsowlo chi sou2al wajed f documentation dyal vLLM
    test_query = "How to configure OpenAI compatible server?"
    print(f"\nKan9elbo 3la: '{test_query}'\n")
    
    # Njbdo a7san 3 dyal l'chunks
    results = retriever.search(test_query, k=3)
    
    # Ntb3o resultats
    for i, res in enumerate(results):
        print(f"--- Top {i+1} ---")
        print(f"Fichier: {res['file_path']}")
        print(f"Index: [{res['first_character_index']} : {res['last_character_index']}]")
        # Ntb3o ghir 150 7erf lowlin mn l'chunk bash terminal may3merch
        print(f"Text: {res['text'][:150]}...\n")