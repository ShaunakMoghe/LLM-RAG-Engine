# LLM RAG Engine

A multi-user, section-aware **Retrieval-Augmented Generation (RAG)** system built with **FastAPI, FAISS, SQLite/SQLAlchemy, Next.js, and Ollama**.
Supports PDF ingestion, hierarchical retrieval (sections → chunks), and real-time Q&A with citations. Designed as a scalable, auditable foundation for enterprise AI.
Note: Development currently in progress and only local deployment/hosting has been tested as detailed below. 

---

## Features

* **Authentication & Multi-tenancy**
  JWT-based login, per-user document isolation, and role management.

* **Document Ingestion**
  PDF upload with SHA-256 deduplication, soft/hard delete, and library view.

* **Indexing & Retrieval**
  Section + chunk embeddings via sentence-transformers, dual FAISS indexes, hierarchical query routing, and diversity filtering.

* **Frontend (Next.js)**
  Document library, indexing progress, and interactive chat UI with citations.

* **Persistence & Logging**
  SQLite + SQLAlchemy schema for users, documents, chunks, and retrieval logs, ensuring auditability.

* **LLM Integration**
  Default: Ollama (Llama 3 8B), with pluggable support for Hugging Face TGI, vLLM, or OpenAI.

---

## Architecture

* **Frontend**
  Next.js

* **Backend**
  FastAPI

* **Vector DB**
  FAISS

* **SQLAlchemy**
  SQLite

---

## Getting Started

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate   # (Windows: .\venv\Scripts\activate)
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

* Frontend: [http://localhost:3000](http://localhost:3000)
* API docs: [http://localhost:8000/api](http://localhost:8000/api)

---

## Example Workflow

1. Upload PDF → stored & deduped (SHA-256).
2. Index → sections + chunks embedded, vectors → FAISS, metadata → DB.
3. Query → hierarchical retrieval (sections → chunks), results → LLM.
4. Answer → returned with doc/page citations; retrieval logged for audit.

---

## Roadmap

* Incremental indexing & multi-doc queries
* Postgres migration for scale
* Extended LLM providers (TGI/vLLM/OpenAI)
* Observability dashboards & admin controls

---

## License

MIT License

---

## Contact

[shaunakmoghe010@gmail.com](mailto:shaunakmoghe010@gmail.com)


