"""Pydantic models for RAG dataset questions and answers."""

from __future__ import annotations

import uuid
from typing import List

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A source reference with file path and character offsets."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question that has not yet been answered."""

    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question that has been answered with sources."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions, answered or unanswered."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Minimal search results for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results that include an answer string."""

    answer: str


class StudentSearchResults(BaseModel):
    """A collection of student search results with k."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Student search results that include answers."""

    search_results: List[MinimalAnswer]
    k: int
