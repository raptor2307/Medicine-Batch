"""
Chroma-backed vector store of known banned/restricted medicines (India + USA).
Seeded once from data/banned_medicines.json. Used as the first lookup layer
before falling back to a live web search via the LLM.
"""

import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from decouple import config

CHROMA_DIR = config("CHROMA_DIR", default="./chroma_db")
DATA_FILE = Path(__file__).parent / "data" / "banned_medicines.json"
COLLECTION_NAME = "banned_medicines"
MATCH_THRESHOLD = config("VECTOR_MATCH_THRESHOLD", default=0.35, cast=float)

_client = None
_collection = None
_embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _doc_text(entry: dict) -> str:
    return (
        f"{entry['name']} ({', '.join(entry['aliases'])}) - {entry['category']}. "
        f"Reason: {entry['reason']}"
    )


def init_vector_store(force_reseed: bool = False):
    """Creates the collection and seeds it from banned_medicines.json if empty."""
    global _collection
    client = _get_client()
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=_embedder
    )

    if force_reseed:
        client.delete_collection(COLLECTION_NAME)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME, embedding_function=_embedder
        )

    if _collection.count() == 0:
        with open(DATA_FILE, "r") as f:
            entries = json.load(f)

        _collection.add(
            ids=[str(i) for i in range(len(entries))],
            documents=[_doc_text(e) for e in entries],
            metadatas=[
                {
                    "name": e["name"],
                    "banned_in": json.dumps(e["banned_in"]),
                    "restricted_in": json.dumps(e.get("restricted_in", [])),
                    "reason": e["reason"],
                    "category": e["category"],
                }
                for e in entries
            ],
        )
    return _collection


def get_collection():
    global _collection
    if _collection is None:
        init_vector_store()
    return _collection


def search_banned_db(query_text: str, top_k: int = 3) -> list[dict]:
    """
    Searches the vector DB for medicines matching the query text.
    Returns a list of matches with similarity distance; empty list if nothing
    clears the similarity threshold (caller should fall back to web search).
    """
    collection = get_collection()
    results = collection.query(query_texts=[query_text], n_results=top_k)

    matches = []
    if not results["ids"] or not results["ids"][0]:
        return matches

    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]  # cosine distance, lower = closer
        if distance <= MATCH_THRESHOLD:
            meta = results["metadatas"][0][i]
            matches.append(
                {
                    "name": meta["name"],
                    "banned_in": json.loads(meta["banned_in"]),
                    "restricted_in": json.loads(meta["restricted_in"]),
                    "reason": meta["reason"],
                    "category": meta["category"],
                    "distance": distance,
                }
            )
    return matches
