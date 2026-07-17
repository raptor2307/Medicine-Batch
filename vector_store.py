"""
Chroma-backed vector store of known banned/restricted medicines (India + USA).
Seeded once from data/banned_medicines.json. Used as the first lookup layer
before falling back to a live web search via the LLM.
"""

import json
import re
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
        name=COLLECTION_NAME,
        embedding_function=_embedder,
        metadata={"hnsw:space": "cosine"},
    )

    # Rebuild collections created before we switched to cosine space and
    # started storing aliases in metadata — their distances/thresholds and
    # exact-match lookups would otherwise be wrong.
    if not force_reseed and _collection.count() > 0:
        space = (_collection.metadata or {}).get("hnsw:space")
        sample = _collection.get(limit=1)
        has_aliases = bool(sample["metadatas"]) and "aliases" in sample["metadatas"][0]
        force_reseed = space != "cosine" or not has_aliases

    if force_reseed:
        client.delete_collection(COLLECTION_NAME)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_embedder,
            metadata={"hnsw:space": "cosine"},
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
                    "aliases": json.dumps(e.get("aliases", [])),
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


def _match_from_meta(meta: dict, distance: float) -> dict:
    return {
        "name": meta["name"],
        "banned_in": json.loads(meta["banned_in"]),
        "restricted_in": json.loads(meta["restricted_in"]),
        "reason": meta["reason"],
        "category": meta["category"],
        "distance": distance,
    }


def _exact_name_matches(query_text: str, collection) -> list[dict]:
    """Whole-word match of any known medicine name/alias inside the query text.
    Catches cases the embedding search misses when the query carries OCR noise."""
    lowered = query_text.lower()
    matches = []
    for meta in collection.get()["metadatas"]:
        candidates = [meta["name"], *json.loads(meta.get("aliases", "[]"))]
        for candidate in candidates:
            # "Analgin (Metamizole / Dipyrone)" -> match on "analgin"
            candidate = candidate.split("(")[0].strip().lower()
            if candidate and re.search(rf"\b{re.escape(candidate)}\b", lowered):
                matches.append(_match_from_meta(meta, 0.0))
                break
    return matches


def search_banned_db(query_text: str, top_k: int = 3) -> list[dict]:
    """
    Searches the banned-medicines DB for the query text: exact name/alias
    matching first, then vector similarity. Returns a list of matches; empty
    list if nothing clears the similarity threshold (caller should fall back
    to web search).
    """
    collection = get_collection()

    exact = _exact_name_matches(query_text, collection)
    if exact:
        return exact

    results = collection.query(query_texts=[query_text], n_results=top_k)

    matches = []
    if not results["ids"] or not results["ids"][0]:
        return matches

    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]  # cosine distance, lower = closer
        if distance <= MATCH_THRESHOLD:
            matches.append(_match_from_meta(results["metadatas"][0][i], distance))
    return matches
