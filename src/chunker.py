from typing import List
from langchain_text_splitters import  RecursiveCharacterTextSplitter, Language

class LangChainChunker:
    def __init__(self, max_chunk_size :int = 2000):
        self.max_chunk_size = max_chunk_size
    

    def chunk_markdown(self, text: str, file_path: str)-> List[dict]:
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = self.max_chunk_size,
            chunk_overlap=0,
            add_start_index=True,
            separators=["\n\n","\n"," ",""]
        )

        docs = splitter.create_documents([text])
        chunks_info = []
        for doc in docs:
            chunk_text = doc.page_content 
            start_index = doc.metadata['start_index']
            end_index = start_index + len(chunk_text)


            chunks_info.append({
                "file_path" : file_path,
                "first_character_index": start_index,
                "last_character_index": end_index,
                "text": chunk_text
            })
        return chunks_info
    
    def chunk_python(self, code :str ,file_path: str) -> List[dict]:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language = Language.PYTHON,
            chunk_size = self.max_chunk_size,
            chunk_overlap=0,
            add_start_index = True,
            # separators = ['\nclass ', '\ndef ', '\n\tdef']
        )

        docs = splitter.create_documents([code])

        chunks_info = []
        for doc in docs:
            chunk_text = doc.page_content
            start_index = doc.metadata['start_index']
            end_index = start_index + len(chunk_text)

            chunks_info.append({
                "file_path" : file_path,
                "first_character_index": start_index,
                "last_character_index": end_index,
                "text": chunk_text
            })
        return chunks_info



# if __name__ == "__main__":

#     test_chunker = LangChainChunker(max_chunk_size=50)

#     print("--- TEST MARKDOWN CHUNKING ---")

#     md_text = "Title 1\n\nThis is the first paragraph.\n\nThis is the second paragraph which is slightly longer."
#     py_text = "def hello():\n    print('Hello World')\n\nclass MyClass:\n    def __init__(self):\n        pass"
#     md_chunks = test_chunker.chunk_markdown(md_text,"docs/test.md")
#     py_chunks = test_chunker.chunk_python(py_text, "docs/test.md")

#     for i, chunk in enumerate(md_chunks):
#         start = chunk['first_character_index']
#         end = chunk['last_character_index']
#         c_text = chunk['text']

#         is_correct = (md_text[start:end] == c_text) 
        
#         print(f"Chunk {i+1}:")
#         print(f"  Start: {start} | End: {end}")
#         print(f"  Valid Index? {is_correct}")
#         print(f"  Text: {repr(c_text)}\n")

#     print("--- TEST PYTHON CHUNKING ---")
#     for i, chunk in enumerate(py_chunks):
#         start = chunk['first_character_index']
#         end = chunk['last_character_index']
#         c_text = chunk['text']

#         is_correct = (md_text[start:end] == c_text) 
        
#         print(f"Chunk {i+1}:")
#         print(f"  Start: {start} | End: {end}")
#         print(f"  Valid Index? {is_correct}")
#         print(f"  Text: {repr(c_text)}\n")