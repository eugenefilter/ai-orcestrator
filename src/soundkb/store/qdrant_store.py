"""Thin Qdrant wrapper: one collection, one named text vector, payload = ``Entry``."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from .embed import MODEL_NAME, VECTOR_SIZE

COLLECTION = "soundkb_entries"
VECTOR = "text"
INDEXES = {
    "kind": models.PayloadSchemaType.KEYWORD,
    "tags": models.PayloadSchemaType.KEYWORD,
    "analysis.features.pitch.f0_steady": models.PayloadSchemaType.FLOAT,
    "analysis.features.meta.duration_s": models.PayloadSchemaType.FLOAT,
}


class ModelMismatch(RuntimeError):
    """The collection was built with a different embedding model."""


def connect(url: str) -> QdrantClient:
    """Open a client; ``:memory:`` works for tests."""
    return QdrantClient(location=url) if url == ":memory:" else QdrantClient(url=url, check_compatibility=False)


def ensure_collection(client: QdrantClient) -> None:
    """Create the collection and payload indexes if missing; refuse to mix embedding models."""
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config={VECTOR: models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)},
            metadata={"embedding_model": MODEL_NAME},
        )
        for field, schema in INDEXES.items():
            client.create_payload_index(COLLECTION, field_name=field, field_schema=schema)
        return
    stored = (client.get_collection(COLLECTION).config.metadata or {}).get("embedding_model")
    if stored and stored != MODEL_NAME:
        raise ModelMismatch(f"Collection uses '{stored}', code uses '{MODEL_NAME}'. Reindex or delete the collection.")


def build_filter(kind: str | None = None, tags: list[str] | None = None, any_tag: bool = False,
                 f0_range: tuple[float, float] | None = None,
                 duration_range: tuple[float, float] | None = None) -> models.Filter | None:
    """Translate search filters into a Qdrant filter."""
    must: list[Any] = []
    if kind:
        must.append(models.FieldCondition(key="kind", match=models.MatchValue(value=kind)))
    if tags and not any_tag:
        must += [models.FieldCondition(key="tags", match=models.MatchValue(value=t)) for t in tags]
    elif tags:
        must.append(models.FieldCondition(key="tags", match=models.MatchAny(any=tags)))
    if f0_range:
        must.append(models.FieldCondition(key="analysis.features.pitch.f0_steady", range=models.Range(gte=f0_range[0], lte=f0_range[1])))
    if duration_range:
        must.append(models.FieldCondition(key="analysis.features.meta.duration_s", range=models.Range(gte=duration_range[0], lte=duration_range[1])))
    return models.Filter(must=must) if must else None


def upsert(client: QdrantClient, point_id: str, vector: list[float], payload: dict[str, Any]) -> None:
    client.upsert(COLLECTION, points=[models.PointStruct(id=point_id, vector={VECTOR: vector}, payload=payload)], wait=True)


def search(client: QdrantClient, vector: list[float], flt: models.Filter | None, limit: int) -> list[tuple[str, float, dict[str, Any]]]:
    hits = client.query_points(COLLECTION, query=vector, using=VECTOR, query_filter=flt, limit=limit, with_payload=True).points
    return [(str(h.id), h.score, h.payload or {}) for h in hits]


def scroll(client: QdrantClient, flt: models.Filter | None, limit: int = 1000) -> list[tuple[str, dict[str, Any]]]:
    points, _ = client.scroll(COLLECTION, scroll_filter=flt, limit=limit, with_payload=True)
    return [(str(p.id), p.payload or {}) for p in points]


def get(client: QdrantClient, point_id: str) -> dict[str, Any] | None:
    found = client.retrieve(COLLECTION, ids=[point_id], with_payload=True)
    return found[0].payload if found else None


def delete(client: QdrantClient, point_id: str) -> None:
    client.delete(COLLECTION, points_selector=models.PointIdsList(points=[point_id]), wait=True)
