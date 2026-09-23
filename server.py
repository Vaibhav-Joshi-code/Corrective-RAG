"""FastAPI + LangServe: open http://localhost:8000/agent/playground."""
from __future__ import annotations

import os
from fastapi import FastAPI, HTTPException
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langserve import add_routes
from langsmith import traceable
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from crag import run_crag

app = FastAPI(
    title="Corrective RAG (CRAG) Agent",
    version="1.0.0",
    description="Self-correcting RAG with retrieval grading and answer validation.",
)


class QueryRequest(BaseModel):
    question: str = Field(
        "Ask a question about the knowledge base.",
        min_length=3,
        description="Ask a question about the knowledge base."
    )

    @field_validator("question")
    @classmethod
    def meaningful_question(cls, value: str) -> str:
        if len(value.strip()) < 3:
            raise ValueError("Enter at least three non-whitespace characters.")
        return value.strip()


class QueryResponse(BaseModel):
    answer: str
    final_status: str
    validation: dict
    retrieval_query: str
    retrieval_attempts: int = Field(description="Additional retrieval cycles after the first.")
    answer_attempts: int = Field(description="Total answer attempts, including the first.")
    document_grades: list[dict]
    relevant_document_ids: list[str]
    correction_reason: str


def invoke_agent(payload: dict, config: RunnableConfig) -> dict:
    """Adapt the question form to the graph and expose useful teaching output."""
    request = QueryRequest.model_validate(payload)
    result = run_crag(request.question, config=config)
    return QueryResponse(
        answer=result.get("answer", ""),
        final_status=result.get("final_status", "unknown"),
        validation=result.get("validation", {}),
        retrieval_query=result.get("retrieval_query", request.question),
        retrieval_attempts=result.get("retrieval_attempts", 0),
        answer_attempts=result.get("answer_attempts", 0),
        document_grades=result.get("document_grades", []),
        relevant_document_ids=[
            str(doc.metadata.get("id", "")) for doc in result.get("relevant_documents", [])
        ],
        correction_reason=result.get("correction_reason", ""),
    ).model_dump()


agent = RunnableLambda(invoke_agent, name="crag_agent").with_types(
    input_type=QueryRequest, output_type=QueryResponse,
)
add_routes(app, agent, path="/agent", playground_type="default")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "langsmith_tracing": os.getenv("LANGSMITH_TRACING", "false"),
    }


@app.post("/crag", response_model=QueryResponse)
@traceable(name="crag_api", run_type="chain")
def crag_endpoint(request: QueryRequest) -> QueryResponse:
    try:
        return QueryResponse(**invoke_agent(request.model_dump(), config={}))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"CRAG execution failed: {type(exc).__name__}: {exc}",
        ) from exc

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        app,
        port=8000,
        log_level="info"
    )
