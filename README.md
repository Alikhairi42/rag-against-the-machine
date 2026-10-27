# RAG Against The Machine

*This project has been created as part of the 42 curriculum by <YOUR_LOGIN>.*

## Description
This project implements a complete Retrieval-Augmented Generation (RAG) pipeline from scratch to answer queries based on a given codebase (vLLM repository). The goal is to ingest documents, index them for fast retrieval, and use a local Large Language Model (Qwen/Qwen3-0.6B) to generate accurate, context-grounded answers without hallucination.

## System Architecture
The RAG pipeline is divided into four main components that interact seamlessly:
1. **Indexer (`indexer.py`)**: Traverses the raw vLLM dataset, identifies `.md` and `.py` files, extracts the text, and passes it to the Chunker. It then builds a search index and saves it to the disk.
2. **Chunker (`chunker.py`)**: Breaks down large files into smaller, manageable pieces (chunks) under 2000 characters while preserving the exact character indices from the original document.
3. **Retriever (`retriever.py`)**: Loads the saved index from the disk and performs a lexical search to find the Top-K most relevant chunks for a given user query.
4. **Generator (`generator.py`)**: Prompts a local LLM (Qwen) with the user's question and the retrieved context chunks to generate a factual, grounded answer.

## Chunking Strategy
Document segmentation is handled using `RecursiveCharacterTextSplitter` from LangChain. 
- **Markdown Files**: Split recursively using separators `["\n\n", "\n", " ", ""]` to ensure paragraphs and sentences stay intact.
- **Python Files**: Split using AST-aware Python separators to preserve classes and functions.
- **Indices Tracking**: The `add_start_index=True` feature is utilized to accurately track the `first_character_index` and `last_character_index`, which is crucial for Moulinette evaluation.

## Retrieval Method
The retrieval mechanism relies on **BM25 (Best Matching 25)**, a highly effective probabilistic Information Retrieval model (an evolution of TF-IDF). 
- The query and the document chunks are tokenized (lowercased and split by whitespace).
- BM25 scores each chunk based on Term Frequency (TF) and Inverse Document Frequency (IDF).
- The system then ranks the chunks in descending order and returns the Top-K results.

## Design Decisions
- **Strict Generation**: The LLM generation parameters are set with `do_sample=False` (Greedy Search) to prevent hallucination and force the model to answer strictly based on the provided context.
- **Data Validation**: `Pydantic` is used extensively to validate all inputs and outputs between the pipeline stages, preventing type errors.
- **Command-Line Interface**: `Python Fire` is used to expose the pipeline through a clean and intuitive CLI.

## Challenges Faced
- **Index Preservation**: Maintaining the exact original character indices after splitting the text was challenging. This was solved by configuring the splitter to track the start index and calculating the end index based on the chunk's length.
- **Memory Management**: Rebuilding the index for every query was inefficient. This was mitigated by persisting the chunks and the BM25 object to disk using Python's `pickle` module.

## Performance Analysis
- **Indexing Time**: The complete ingestion of the vLLM codebase takes under 1 minute.
- **Retrieval Quality**: The BM25 algorithm accurately retrieves the exact configuration commands and documentation sections, easily passing the required Recall@5 thresholds (80% on docs, 50% on code).

## Instructions
### Installation
Ensure you have `uv` installed, then run:
```bash
make install


### Execution (End-to-End Pipeline)
1. **Index the corpus**:
```bash
uv run python -m src index --max_chunk_size 2000
```
2. **Search a dataset**:
```bash
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
```
3. **Generate answers for a dataset**:
```bash
uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

## Example Usage (Single Query)
Search the index directly from the terminal:
```bash
uv run python -m src search "How to configure OpenAI server?" --k 3
```
Ask a question directly to the LLM:
```bash
uv run python -m src answer "How to configure OpenAI server?" --k 3
```

## Resources
- [What is RAG? (IBM)](https://www.youtube.com/watch?v=T-D1OfcDW1M)
- [Practical BM25 - ElasticSearch](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)
- **AI Usage:** Generative AI was used strictly as a sparring partner to understand the mathematical concepts of BM25 and to assist in debugging specific Python exception tracebacks during the development phase. The core logic and integration were implemented manually.
