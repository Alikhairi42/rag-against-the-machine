import fire
import json
from pathlib import Path
from tqdm import tqdm
import os
from models import MinimalSource, MinimalSearchResults, StudentSearchResults

from indexer import CodebaseIndexer
from retriever import BM25Retriever

class RAGCLI:
    def index(self, max_chunk_size: int = 2000):
        raw_path = "data/raw/vllm-0.10.1"
        print(f"start Indexing dyal {raw_path} ...")
        indexer = CodebaseIndexer(max_chunk_size=max_chunk_size)
        indexer.ingest_directory(raw_path)
        indexer.token_and_build_bm25()
        indexer.save_index()
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 5):
        retriever = BM25Retriever()
        results = retriever.search(query, k=k)
        print(f"\nResultats dyal: '{query}'")
        for i, res in enumerate(results):
            print(f"--- Top {i+1} ---")
            print(f"File: {res['file_path']}")
            print(f"Index: [{res['first_character_index']} : {res['last_character_index']}]")
            print(f"Text: {res['text'][:100]}...\n")
    def search_dataset(self, dataset_path: str, save_directory: str, k: int = 5):
        if not os.path.exists(dataset_path):
            print(f"Erreur: L'fichier {dataset_path} notfon!")
            return
if __name__ == "__main__":
    fire.Fire(RAGCLI)
