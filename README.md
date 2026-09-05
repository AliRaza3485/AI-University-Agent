# AI University Agent

RAG-based AI assistant for **University of Education, Lahore** that answers questions from official prospectus documents.

## Quick Start

```bash
# 1. Clone & enter
git clone <repo-url>
cd ai-university-agent

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your values

# 5. Run the API server
uvicorn app.main:app --reload

# 6. Run tests
pytest
```

## Project Structure

```
ai-university-agent/
├── app/
│   ├── agents/          # AI agent logic
│   ├── api/             # FastAPI route handlers
│   ├── config/          # Centralized settings (pydantic-settings)
│   ├── database/        # DB models & session management
│   ├── embeddings/      # Text embedding generation
│   ├── evaluation/      # RAG quality evaluation
│   ├── ingestion/       # PDF parsing & text cleaning
│   ├── memory/          # Conversation history
│   ├── models/          # Pydantic schemas
│   ├── reranking/       # Result reranking
│   ├── retrieval/       # Vector search & retrieval
│   ├── services/        # Business logic layer
│   ├── tools/           # Agent tool definitions
│   ├── utils/           # Shared helpers
│   └── main.py          # FastAPI app entrypoint
├── data/
│   ├── raw_documents/   # Source PDFs (git-ignored)
│   ├── processed/       # Cleaned & chunked output
│   ├── embeddings/      # Generated vector embeddings
│   └── test_dataset/    # Evaluation Q&A pairs
├── notebooks/           # Exploration & verification notebooks
├── tests/               # Pytest test suite
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Tech Stack

- **Python 3.11+**
- **FastAPI** — async API framework
- **PyMuPDF** — PDF text extraction
- **Pydantic v2** — data validation & settings
- **SQLAlchemy 2** — database ORM
- **PostgreSQL 16** — relational database
- **Docker** — containerized deployment

## Development

```bash
# Run tests
pytest

# Run tests with output
pytest -s

# Run a specific test file
pytest tests/unit/test_cleaner.py
```

## Docker

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down
```

## License

MIT
