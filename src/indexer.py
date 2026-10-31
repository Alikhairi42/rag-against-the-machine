from typing import List ,Dict 
from pathlib import Path
from chunker import LangChainChunker
from rank_bm25 import BM25Okapi
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from incremental import IncrementalTracker

class CodebaseIndexer:
    def __init__(self, max_chunk_size: int = 2000):
        self.chunker = LangChainChunker(max_chunk_size=max_chunk_size)
        self.chunks : List[Dict] = []
        print("Kan-loadiw Embedding Model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None

    def ingest_directory(self,raw_data_path: str):
        print(f"read data in this path : {raw_data_path}")

        base_path = Path(raw_data_path)
        lst  = base_path.rglob("*")

        for file_path in lst:
            if file_path.name.startswith("."):
                continue
            ex = file_path.suffix.lower()

            if ex in [".md", ".py", ".txt"]:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    if ex == ".md":
                        new_chunks = self.chunker.chunk_markdown(text, file_path)
                    elif ex == ".txt":
                        new_chunks = self.chunker.chunk_markdown(text, file_path)
                    else:
                        new_chunks = self.chunker.chunk_python(text, file_path)
                    
                    self.chunks.extend(new_chunks)
                    

                except Exception as e:
                    pass

    def token_and_build_bm25(self):
        print("start tokenized ...")

        tokenized_docs = []

        for chunk in self.chunks:
            token = chunk['text'].lower().split()
            tokenized_docs.append(token)
        
        print("index BM 25 ....")
        self.bm25 = BM25Okapi(tokenized_docs)

    
    def save_index(self, save_dir: str = "data/processed"):
        print("save index ....")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        chunks_path = Path(save_dir)

        with open(f"{chunks_path}/chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)


        bm25_path = Path(save_dir)
        with open(f"{bm25_path}/bm25_index.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        embeddings_path = Path(save_dir)
        with open(f"{embeddings_path}/embeddings.pkl", "wb") as f:
            pickle.dump(self.embeddings,f)
        print("save index vector")

    def build_vector_index(self):
        print(" Vectors (Embeddings)...")
        for chunk in self.chunks:
            texts = chunk['text']
        self.embeddings =self.embedding_model.encode(texts, show_progress_bar=True)
        print("vector index wajed!")

        






if __name__ == "__main__":
    
    index = CodebaseIndexer(max_chunk_size=2000)

    raw_path = "data/raw/vllm-0.10.1"

    index.ingest_directory(raw_path)
    index.token_and_build_bm25()
    index.save_index()




