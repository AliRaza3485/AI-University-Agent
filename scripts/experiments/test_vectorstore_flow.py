"""
Manual smoke test: ingest a real PDF, embed all its chunks, store
them in Qdrant, then run a sample search. Not a pytest test - this is
for eyeballing that the full flow works end-to-end against the real
Qdrant Cloud instance.

Run with:
    python scripts/experiments/test_vectorstore_flow.py
"""

from app.embeddings.embedder import embed_chunks, embed_texts
from app.ingestion.pipeline import ingest_document
from app.vectorstore.store import search, upsert_embedded_chunks

PDF_PATH = "data/raw_documents/Prospectus - FALL 2026 (29-07-2026).pdf"


def main() -> None:
    print(f"1. Ingesting: {PDF_PATH}")
    chunks = ingest_document(PDF_PATH)
    print(f"   -> {len(chunks)} total chunks in the document")

    print(f"2. Embedding all {len(chunks)} chunks (this will take a minute)...")
    embedded = embed_chunks(chunks)
    print(
        f"   -> {len(embedded)} embeddings generated (dimension={len(embedded[0].embedding)})"
    )

    print("3. Upserting into Qdrant...")
    count = upsert_embedded_chunks(embedded)
    print(f"   -> {count} points stored")

    query = "What are the admission requirements?"
    print(f"4. Searching for: {query!r}")
    query_vector = embed_texts([query])[0]
    results = search(query_vector, top_k=3)

    print(f"   -> top {len(results)} results:")
    for i, r in enumerate(results, start=1):
        print(f"      [{i}] score={r['score']:.3f} page={r['page_numbers']}")
        print(f"          {r['text'][:120]}...")


if __name__ == "__main__":
    main()
