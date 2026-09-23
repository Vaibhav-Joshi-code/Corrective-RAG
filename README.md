# 🔄 Corrective RAG (CRAG) — Self-Correcting RAG Agent

> A complete Corrective RAG pipeline that grades retrieval quality, rewrites weak queries, retrieves again, generates a grounded answer, validates grounding/completeness/hallucination risk, and automatically retries when validation fails.

Built with **LangGraph, LangChain, Azure OpenAI, Qdrant, FastAPI, LangServe, and LangSmith**.

## Qdrant setup and playground

Retrieval now uses a persistent **Qdrant vector database**, connected using the existing
`QDRANT_URL` and `QDRANT_API_KEY` settings in `.env`. Azure OpenAI creates embeddings
and provides the chat model; Qdrant stores vectors, source text, and metadata and runs
similarity searches. LangGraph still performs relevance grading, rewriting, generation,
validation, and bounded retries.

Use `requirements.txt` for the current setup:

```powershell
python -m pip install -r requirements.txt

python -X utf8 -m uvicorn server:app --reload --port 8000
```

Open **http://localhost:8000/agent/playground**. 

```dotenv
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=corrective_rag_crag
```

Use the actual cluster URL and API key supplied by Qdrant. Collection name is optional;
the default is `corrective_rag_crag`. Uppercase names are recommended; the adapter also
accepts lowercase `qdrant_url` and `qdrant_api_key`. A self-hosted local instance can use
an empty API key if its authentication settings allow it. Credentials stay in `.env`.

### Storage and startup behaviour

- `qdrant_store.py` uses the official `qdrant-client`; no `langchain-qdrant` dependency is required.
- Startup embeds the sample documents through Azure and creates the collection if absent.
  Vector dimension comes from the actual embedding output; distance is cosine.
- Source IDs become deterministic UUID point IDs. Repeated startup **upserts** the same
  points, updating their vectors/text/metadata without appending duplicate documents.
- Existing collections are checked for the expected dimension and cosine metric.
  An incompatible collection produces an error; it is never automatically recreated or deleted.
  Choose a new collection name when switching embedding deployments or incompatible schemas.
- Each point's payload contains `page_content` and `metadata` (including `id` and `topic`).
  Results are converted back to LangChain `Document` objects for the existing grader.
- Searches are restricted to the current knowledge-base point IDs. Removed source entries
  and unrelated points are excluded from retrieval but are not deleted from Qdrant.
- Qdrant data survives a FastAPI restart. This small teaching app still re-embeds and upserts
  the knowledge base on each startup/reload; it does not implement incremental ingestion.
  Persistent storage does not imply zero embedding cost on restart.
- Live startup therefore needs Azure embedding access **and** Qdrant connectivity and
  permission to create/write the selected collection. Keep a dedicated collection for this lesson.

### Verification

The 15 graph/playground tests use model/store doubles. Eight additional tests use real
Qdrant **local mode** with deterministic fake embeddings, checking search/payloads,
idempotent updates, persisted data after reopening, scope filtering, and incompatible
collections. These 23 offline checks require no cloud credentials and do not certify
connectivity or model quality on the live Azure/Qdrant services.

---

## 🎯 Project Coverage

The implementation directly maps to the requested flow:

```text
Question
   ↓
Retrieve
   ↓
Grade Documents
   ↓
Relevant?
 ┌─┴─────────────┐
No               Yes
↓                  ↓
Rewrite Query    Generate
↓                  ↓
Retrieve Again   Validate
                   ↓
              Answer Valid?
              ┌────┴─────┐
             No          Yes
             ↓             ↓
       Retry Retrieval   Finish
             ↓
          Retrieve
```

### Requirement-to-Code Matrix

| Requirement | Implementation |
|---|---|
| Complete CRAG pipeline | `crag.py` |
| Retrieve | `retrieve()` |
| Grade documents | `grade_documents()` |
| Relevant / irrelevant decision | Structured `grader_llm` |
| Correct / rewrite | `rewrite_query()` |
| Retrieve again | `rewrite_query → retrieve` |
| Generate | `generate()` |
| Validate | `validate_answer()` |
| Retrieval relevance | `document_grades`, `relevant_documents` |
| Answer grounding | Validator `grounded` |
| Answer completeness | Validator `complete` |
| Hallucination risk | Validator `hallucination_risk` |
| Another retrieval cycle | `retry_retrieval()` |
| Automatic retry | Conditional edge after validation |
| Maximum retry protection | `MAX_RETRIEVAL_RETRIES`, `MAX_VALIDATION_RETRIES` |
| LangGraph | `StateGraph` |
| Conditional routing | `add_conditional_edges()` |
| Retrieval | `QdrantStore` in `qdrant_store.py`, backed by Qdrant |
| Predefined dataset | `knowledge_base.py` |
| LangSmith | Explicit `@traceable` |
| API | `server.py` |

---

# 🧠 What is CRAG?

Traditional RAG:

```text
Question → Retrieve → Generate
```

CRAG does not blindly trust the retriever:

```text
Question
   ↓
Retrieve
   ↓
Grade Documents
   ↓
Are they relevant?
   ├── Yes → Generate
   └── No  → Rewrite Query → Retrieve Again
```

This project extends CRAG with answer self-validation:

```text
Generate
   ↓
Validate
   ↓
Grounded?
Complete?
Hallucination risk low?
   │
 ┌─┴──────┐
Yes       No
 │         │
END    Retry Retrieval
          ↓
       Generate Again
```

So the project implements both **corrective retrieval** and **self-correcting answer generation**.

---

# 🏗️ Architecture

```text
                         ┌──────────────────┐
                         │     Question     │
                         └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │     Retrieve     │
                         └────────┬─────────┘
                                  ↓
                     ┌────────────────────────┐
                     │  Grade Documents       │
                     │ Relevant / Irrelevant  │
                     └───────────┬────────────┘
                                 ↓
                         Documents relevant?
                           /                                      No               Yes
                         ↓                 ↓
                ┌────────────────┐   ┌──────────┐
                │ Rewrite Query  │   │ Generate │
                └───────┬────────┘   └────┬─────┘
                        ↓                  ↓
                  Retrieve Again       Validate
                        ↑                  ↓
                        │       ┌──────────┴──────────┐
                        │       │                     │
                        │      PASS                 FAIL
                        │       │                     │
                        │       ↓                     ↓
                        │     Finish          Retry Retrieval
                        │                             │
                        └─────────────────────────────┘
```

---

# 📂 Project Structure

```text
corrective-rag-crag-agent/
│
├── crag.py
│   ├── Azure OpenAI model
│   ├── embeddings
│   ├── vector store
│   ├── CRAGState
│   ├── retrieve
│   ├── grade_documents
│   ├── rewrite_query
│   ├── generate
│   ├── validate_answer
│   ├── retry_retrieval
│   ├── maximum retry protection
│   ├── StateGraph
│   └── explicit LangSmith tracing
│
├── knowledge_base.py
│   └── predefined documents
│
├── server.py
│   ├── FastAPI
│   ├── POST /crag
│   └── GET /health
│
├── smoke_test.py
├── qdrant_store.py
├── test_qdrant_store.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# 🔎 Retrieval

The project uses a persistent Qdrant collection through the `QdrantStore` adapter.

The predefined documents are converted into LangChain `Document` objects, embedded
with Azure OpenAI, and upserted into Qdrant. Each query is embedded with the same
deployment; Qdrant's `query_points` returns the nearest candidate passages. The
following interface stays the same for the graph:

```python
docs = vector_store.similarity_search(
    query,
    k=TOP_K,
)
```

Default:

```env
TOP_K=4
```

The local knowledge base contains both relevant technical information and distractor documents so the grading/correction logic is observable.

---

# 🧑‍⚖️ Document Relevance Grading

Every retrieved document is explicitly graded.

The grader returns structured data:

```json
{
  "relevant": true,
  "reason": "The document directly supports the question."
}
```

or:

```json
{
  "relevant": false,
  "reason": "The document does not contain evidence needed for this question."
}
```

Only documents with:

```text
relevant = true
```

are passed to the generator.

This satisfies the requirement that retrieved documents must be classified as **relevant/irrelevant**.

---

# ✏️ Corrective Query Rewriting

When no retrieved document is relevant:

```text
Retrieve
   ↓
Grade
   ↓
No Relevant Documents
   ↓
Rewrite Query
   ↓
Retrieve Again
```

The rewriting model receives:

- original question,
- previous document grades.

It produces a more targeted search query.

Example:

```text
Original:
Why is my API failing?

Rewritten:
HTTP 500 API failure server-side error troubleshooting
```

The rewritten query is then sent to the retriever.

---

# 🔁 Maximum Retrieval Retry

The corrective retrieval loop has a hard limit:

```env
MAX_RETRIEVAL_RETRIES=2
```

This prevents:

```text
Retrieve → Grade → Rewrite → Retrieve → Grade → Rewrite → ...
```

from becoming infinite.

After the configured limit, the graph proceeds to generation, where the system can transparently report insufficient evidence instead of fabricating an answer.

---

# ✍️ Answer Generation

The generator receives only documents that passed relevance grading.

The prompt explicitly requires:

```text
Use only supplied documents.
Do not introduce unsupported facts.
If context does not fully answer the question,
say what is missing.
```

This creates:

```text
Retrieved Documents
       ↓
Relevance Grader
       ↓
Relevant Documents
       ↓
Answer Generator
```

---

# 🔍 Answer Validation

After generation, the answer is evaluated by a separate structured validator.

It checks:

### Grounding

Is the answer supported by the retrieved context?

```text
grounded = true / false
```

### Completeness

Does the answer address the question?

```text
complete = true / false
```

### Hallucination Risk

```text
low / medium / high
```

### Overall Result

```text
pass_validation = true / false
```

The answer passes only when:

```text
grounded = true
AND
complete = true
AND
hallucination_risk = low
```

---

# 🔄 Failed Validation → Automatic Retry

If validation fails:

```text
Generate
   ↓
Validate
   ↓
FAIL
   ↓
Retry Retrieval
   ↓
New Retrieval Query
   ↓
Retrieve
   ↓
Grade
   ↓
Generate Again
   ↓
Validate Again
```

This is a real LangGraph conditional branch, not just a message printed to the console.

---

# 🛑 Maximum Validation Retry

The answer correction loop is also bounded:

```env
MAX_VALIDATION_RETRIES=2
```

This protects the application from an infinite self-correction loop.

If the answer still cannot pass validation after the permitted attempts, the graph terminates with:

```text
completed_with_validation_warning
```

rather than continuing forever.

---

# 🧠 Why Two Correction Mechanisms?

There are two different failure modes.

## Retrieval failure

```text
Evidence is poor
       ↓
Rewrite query
       ↓
Retrieve again
```

## Answer failure

```text
Evidence exists
       ↓
Answer generated
       ↓
Answer fails validation
       ↓
Retrieve additional/targeted evidence
       ↓
Generate again
```

Separating these makes the implementation clear for code review and LangSmith tracing.

---

# 🧪 Successful Generation Example

Run:

```text
What does HTTP 500 mean?
```

Expected conceptual flow:

```text
Question
 ↓
Retrieve
 ↓
Grade
 ↓
http_500 = Relevant
 ↓
Generate
 ↓
Validate
 ↓
Grounded = True
Complete = True
Hallucination Risk = Low
 ↓
Finish
```

---

# 🧪 Corrective Retrieval Example

Run:

```text
What should I investigate when an API returns an internal server error?
```

The initial retrieval can contain a mixture of useful and distracting documents.

The grader explicitly evaluates each one.

If the useful evidence is insufficient:

```text
Grade
 ↓
No sufficient evidence
 ↓
Rewrite Query
 ↓
Retrieve Again
 ↓
Grade Again
```

This demonstrates the core CRAG correction behavior.

---

# ❌ Insufficient-Evidence Example

Try a question about something outside the local knowledge base:

```text
Explain the internal architecture of QuantumDB-X.
```

The system should not invent a detailed answer.

If relevant evidence cannot be found after the allowed retrieval attempts, the generator returns an explicit insufficient-evidence response.

This demonstrates the anti-hallucination behavior expected from a self-correcting RAG system.

---

# 🌐 FastAPI

Run:

```powershell
uvicorn server:app --reload --port 8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Endpoint:

```text
POST /crag
```

Request:

```json
{
  "question": "What does HTTP 500 mean?"
}
```

Response:

```json
{
  "answer": "...",
  "final_status": "validated",
  "validation": {
    "grounded": true,
    "complete": true,
    "hallucination_risk": "low",
    "pass_validation": true,
    "reason": "..."
  }
}
```

---

# ▶️ Installation

```powershell
python -m venv .venv
.venv\Scriptsctivate
pip install -r requirements.txt
```

Copy:

```text
.env.example
```

to:

```text
.env
```

Configure:

```env
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
OPENAI_API_VERSION=your_api_version
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_EMBEDDING_VERSION=your_embedding_api_version
QDRANT_URL=https://your-cluster.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=
```

`AZURE_OPENAI_EMBEDDING_VERSION` can be omitted or blank to use `OPENAI_API_VERSION`.

LangSmith:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=corrective-rag-crag-agent
```

Retry settings:

```env
TOP_K=4
MAX_RETRIEVAL_RETRIES=2
MAX_VALIDATION_RETRIES=2
```

---


# 🔭 LangSmith Tracing

LangSmith is deliberately wired explicitly into the source.

Main pipeline:

```python
@traceable(name="crag_pipeline", run_type="chain")
def run_crag(question: str) -> dict:
    ...
```

FastAPI endpoint:

```python
@traceable(name="crag_api", run_type="chain")
def crag_endpoint(...):
    ...
```

Environment:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=corrective-rag-crag-agent
```

This avoids the ambiguity that can occur when tracing is only implied through environment variables.

---

# 📊 Expected LangSmith Traces

Successful path:

```text
crag_pipeline
      ↓
retrieve
      ↓
grade_documents
      ↓
generate
      ↓
validate_answer
      ↓
finish
```

Corrective retrieval path:

```text
crag_pipeline
      ↓
retrieve
      ↓
grade_documents
      ↓
rewrite_query
      ↓
retrieve
      ↓
grade_documents
      ↓
generate
      ↓
validate_answer
      ↓
finish
```

Self-correction path:

```text
generate
   ↓
validate_answer
   ↓
FAIL
   ↓
retry_retrieval
   ↓
retrieve
   ↓
grade_documents
   ↓
generate
   ↓
validate_answer
   ↓
PASS
   ↓
finish
```

The actual trace path depends on the query and model decisions.

---

# 🔐 Design Decisions

### Why Qdrant?

Qdrant provides vector similarity search, payload storage, filtering, and persistence
outside the Python process. Its cloud and self-hosted options and official Python client
fit this project's existing Qdrant connection settings. We can show collections, points,
vectors, payloads, and repeatable upserts in the practical lesson.

It is an architectural choice, not a claim that Qdrant is always the fastest or that
eleven sample passages require a hosted database. Pinecone, Weaviate, Milvus/Zilliz,
Chroma, and PostgreSQL with pgvector are other options; Elasticsearch/OpenSearch and
Azure AI Search also support vector retrieval. FAISS is a similarity-search library,
not by itself a managed vector database.

Qdrant selects candidate evidence. It does not create the embeddings in this project,
judge document relevance, write answers, or guarantee that generated answers are true.

### Why structured graders?

Free-form grading is difficult to route reliably.

The document grader therefore returns:

```text
relevant
reason
```

The answer validator returns:

```text
grounded
complete
hallucination_risk
pass_validation
reason
```

This makes graph routing explicit.

### Why explicit LangSmith tracing?

The previous task's common tracing weakness is avoided here.

The pipeline itself contains:

```python
@traceable(...)
```

and the API endpoint is also explicitly traced.

---

# 🏁 Final Architecture Summary

```text
                       USER QUESTION
                             │
                             ▼
                         RETRIEVE
                             │
                             ▼
                    GRADE DOCUMENTS
                             │
                 ┌───────────┴───────────┐
                 │                       │
            IRRELEVANT                RELEVANT
                 │                       │
                 ▼                       ▼
          REWRITE QUERY              GENERATE
                 │                       │
                 ▼                       ▼
          RETRIEVE AGAIN             VALIDATE
                                         │
                              ┌──────────┴──────────┐
                              │                     │
                            PASS                  FAIL
                              │                     │
                              ▼                     ▼
                            FINISH           RETRY RETRIEVAL
                                                    │
                                                    ▼
                                                RETRIEVE
```

> **Do not blindly trust retrieval, and do not blindly trust generation.**

That is the core principle demonstrated by this project.
