from typing import List ,Truple
from langchain_text_splitters import  RecursiveCharacterTextSplitter, Language

class LangChainChunker:
    def __init__(self, max_chunk_size :int = 2000):
        self.max_chunk_size = max_chunk_size
    

    def chunk_markdown(self, text: str, file_path:str)-> List[dict]:

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.max_chunk_size,
            chunk_overlap=0,
            add_start_index=True,
            separators=["\n\n","\n"," "," "]
        )

        