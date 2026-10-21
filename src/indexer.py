from typing import List ,Dict 
from pathlib import Path
from chunker import LangChainChunker
class CodebaseIndexer:
    def __init__(self, max_chunk_size: int = 2000):
        self.chunker = LangChainChunker(max_chunk_size=max_chunk_size)
        self.chunks : List[Dict] = []


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

if __name__ == "__main__":
    
    index = CodebaseIndexer(max_chunk_size=2000)

    raw_path = "data/raw/vllm-0.10.1"

    index.ingest_directory(raw_path)




