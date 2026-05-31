from pydantic import BaseModel, Field
from typing import List


class IngestResponse(BaseModel):
    message: str
    filename: str
    chunks_stored: int


class ChatRequest(BaseModel):
    question: str  
    model: str = Field(default="gemini-2.5-flash")
   
    filter_source: str | None = None


class SourceChunk(BaseModel):
    page_number: int
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


class ErrorResponse(BaseModel):
    error: str
    message: str