"""FastAPI application for RAG codebase search and QA."""

from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel, Field

from generator import AnswerGenerator
from model import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
)
from retriever import Retriever

app = FastAPI(
    title="RAG Against The Machine API",
    description=(
        "Local HTTP API for codebase search"
        " and question answering"
    ),
)

retriever = Retriever()
generator = AnswerGenerator()


class QueryData(BaseModel):
    """Request body for search and answer endpoints."""

    query: str = Field(
        ..., example="How to configure OpenAI server?"
    )
    k: int = Field(default=5, example=3)


def execute_rag(
    data: QueryData,
    is_answer: bool = False,
) -> Union[MinimalSearchResults, MinimalAnswer]:
    """Run hybrid search and optionally generate an answer.

    Args:
        data: The incoming query payload.
        is_answer: If True, also generate an answer.

    Returns:
        Search results or a full answer depending on
        *is_answer*.
    """
    results = retriever.hybrid_search(
        data.query, k=data.k
    )

    sources = [
        MinimalSource(
            file_path=str(c["file_path"]),
            first_character_index=c[
                "first_character_index"
            ],
            last_character_index=c[
                "last_character_index"
            ],
        )
        for c in results
    ]

    if not is_answer:
        return MinimalSearchResults(
            question_id="api-search",
            question=data.query,
            retrieved_sources=sources,
        )

    answer_text = generator.generate(
        data.query, results
    )
    return MinimalAnswer(
        question_id="api-answer",
        question=data.query,
        retrieved_sources=sources,
        answer=answer_text,
    )


@app.post(
    "/search", response_model=MinimalSearchResults
)
def search_endpoint(
    search_req: QueryData,
) -> MinimalSearchResults:
    """Return ranked source chunks for a query."""
    return execute_rag(search_req, is_answer=False)


@app.post("/answer", response_model=MinimalAnswer)
def answer_endpoint(
    answer_req: QueryData,
) -> MinimalAnswer:
    """Return an LLM-generated answer with sources."""
    return execute_rag(answer_req, is_answer=True)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a simple health-check response."""
    return {
        "status": "ok",
        "message": "RAG API is running!",
    }
