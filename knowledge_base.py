"""Predefined knowledge base for the CRAG demonstration."""

KNOWLEDGE_BASE = [
    {
        "id": "python_gil",
        "topic": "Python",
        "content": (
            "Python's Global Interpreter Lock (GIL) in CPython historically allows "
            "only one thread at a time to execute Python bytecode within a process. "
            "CPU-bound workloads may therefore benefit from multiprocessing or other "
            "approaches that use multiple processes."
        ),
    },
    {
        "id": "python_async",
        "topic": "Python",
        "content": (
            "Python asyncio is useful for concurrent I/O-bound workloads. It uses an "
            "event loop and cooperative scheduling rather than making CPU-bound "
            "Python bytecode execute concurrently on multiple threads."
        ),
    },
    {
        "id": "http_500",
        "topic": "HTTP",
        "content": (
            "HTTP 500 Internal Server Error indicates that the server encountered "
            "an unexpected condition that prevented it from fulfilling the request. "
            "Application logs and server-side dependencies should be inspected."
        ),
    },
    {
        "id": "http_404",
        "topic": "HTTP",
        "content": (
            "HTTP 404 Not Found indicates that the server cannot find the requested "
            "resource. It commonly occurs when a URL or route does not exist."
        ),
    },
    {
        "id": "http_401",
        "topic": "HTTP",
        "content": (
            "HTTP 401 Unauthorized is used when authentication credentials are "
            "missing or invalid for a protected resource."
        ),
    },
    {
        "id": "database_pool",
        "topic": "Databases",
        "content": (
            "Database connection pool exhaustion occurs when all available "
            "connections are occupied. New requests may wait and eventually time "
            "out. Investigation should include long-running queries, connection "
            "leaks, pool sizing, and connection lifecycle management."
        ),
    },
    {
        "id": "docker_restart",
        "topic": "Docker",
        "content": (
            "A Docker container restart loop can be investigated by checking "
            "container logs, health checks, exit codes, environment variables, "
            "startup commands, and exposed ports."
        ),
    },
    {
        "id": "rag_grounding",
        "topic": "RAG",
        "content": (
            "Grounded RAG answers should be supported by retrieved context. A "
            "validator can compare the generated answer with retrieved documents "
            "and flag unsupported claims."
        ),
    },
    {
        "id": "rag_retrieval",
        "topic": "RAG",
        "content": (
            "Retrieval quality can be improved by grading retrieved documents. "
            "When documents are irrelevant, a corrective RAG workflow can rewrite "
            "the query and retrieve again."
        ),
    },
    {
        "id": "cloud_storage",
        "topic": "Cloud",
        "content": (
            "Object storage is designed for storing objects such as files and blobs. "
            "It is commonly used for backups, media, data lakes, and static assets."
        ),
    },
    {
        "id": "fastapi",
        "topic": "FastAPI",
        "content": (
            "FastAPI is a Python web framework commonly used to build APIs. It "
            "provides request validation using Pydantic models and generates "
            "OpenAPI documentation."
        ),
    },
]
