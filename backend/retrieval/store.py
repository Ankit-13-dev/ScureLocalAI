from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from backend.config import Settings, ensure_storage_dirs, get_settings
from backend.ingestion.chunker import Chunk


class ChromaStore:
    def __init__(self, settings: Settings | None = None) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self.settings = settings or get_settings()
        ensure_storage_dirs(self.settings)
        self.client = chromadb.PersistentClient(
            path=str(self.settings.storage.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.settings.storage.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        # Pictures cut out of PDFs live in their own collection so text retrieval is never polluted.
        self.figure_collection = self.client.get_or_create_collection(
            name=f"{self.settings.storage.chroma_collection}_figures",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        doc_id: str,
        filename: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        ingested_at = datetime.now(timezone.utc).isoformat()
        ids = [f"{doc_id}:{chunk['chunk_index']}" for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "page": int(chunk["page"]),
                "chunk_index": int(chunk["chunk_index"]),
                "ingested_at": ingested_at,
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=max(1, k),
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[dict[str, Any]] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            chunks.append(
                {
                    "text": text,
                    "filename": metadata.get("filename", ""),
                    "doc_id": metadata.get("doc_id", ""),
                    "page": metadata.get("page"),
                    "chunk_index": metadata.get("chunk_index"),
                    "distance": distance,
                }
            )
        return chunks

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})
        self.delete_figures(doc_id)

    # ------------------------------------------------------------------ figures
    def add_figures(
        self,
        doc_id: str,
        filename: str,
        figures: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        """Store one record per picture. ``figures`` items come from ingestion.figures."""
        if not figures:
            return
        if len(figures) != len(embeddings):
            raise ValueError("figures and embeddings must have the same length")

        ingested_at = datetime.now(timezone.utc).isoformat()
        self.figure_collection.upsert(
            ids=[f"{doc_id}:{fig['figure_id']}" for fig in figures],
            documents=[str(fig["search_text"]) for fig in figures],
            embeddings=embeddings,
            metadatas=[
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "figure_id": str(fig["figure_id"]),
                    "page": int(fig["page"]),
                    "kind": str(fig.get("kind", "unknown")),
                    "caption": str(fig.get("caption") or ""),
                    "description": str(fig.get("description") or ""),
                    "fig_number": int(fig.get("fig_number") or -1),
                    "crop_file": str(fig["crop_file"]),
                    "bbox": json.dumps([round(float(v), 2) for v in fig.get("bbox", ())]),
                    "source": str(fig.get("source", "")),
                    "confidence": float(fig.get("confidence", 0.0)),
                    "ingested_at": ingested_at,
                }
                for fig in figures
            ],
        )

    @staticmethod
    def _figure_record(text: str, metadata: dict[str, Any] | None, distance: float | None) -> dict[str, Any]:
        metadata = metadata or {}
        try:
            bbox = json.loads(metadata.get("bbox") or "[]")
        except ValueError:
            bbox = []
        number = metadata.get("fig_number", -1)
        return {
            "text": text,
            "doc_id": metadata.get("doc_id", ""),
            "filename": metadata.get("filename", ""),
            "figure_id": metadata.get("figure_id", ""),
            "page": metadata.get("page"),
            "kind": metadata.get("kind", "unknown"),
            "caption": metadata.get("caption", ""),
            "description": metadata.get("description", ""),
            "fig_number": None if number in (None, -1) else int(number),
            "crop_file": metadata.get("crop_file", ""),
            "bbox": bbox,
            "distance": distance,
        }

    def query_figures(self, embedding: list[float], k: int) -> list[dict[str, Any]]:
        total = self.figure_collection.count()
        if total == 0:
            return []
        result = self.figure_collection.query(
            query_embeddings=[embedding],
            n_results=max(1, min(k, total)),
            include=["documents", "metadatas", "distances"],
        )
        return [
            self._figure_record(text, metadata, distance)
            for text, metadata, distance in zip(
                result.get("documents", [[]])[0],
                result.get("metadatas", [[]])[0],
                result.get("distances", [[]])[0],
            )
        ]

    def figures_by_number(self, number: int) -> list[dict[str, Any]]:
        """Exact lookup for questions like "show me Fig. 3" (no embedding needed)."""
        result = self.figure_collection.get(where={"fig_number": int(number)}, include=["documents", "metadatas"])
        return [
            self._figure_record(text, metadata, None)
            for text, metadata in zip(result.get("documents", []) or [], result.get("metadatas", []) or [])
        ]

    def list_figures(self, doc_id: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
        if doc_id:
            kwargs["where"] = {"doc_id": doc_id}
        result = self.figure_collection.get(**kwargs)
        records = [
            self._figure_record(text, metadata, None)
            for text, metadata in zip(result.get("documents", []) or [], result.get("metadatas", []) or [])
        ]
        records.sort(key=lambda r: (str(r["filename"]), r["page"] or 0, str(r["figure_id"])))
        return records[:limit] if limit else records

    def figure_counts(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for metadata in self.figure_collection.get(include=["metadatas"]).get("metadatas", []) or []:
            if metadata and metadata.get("doc_id"):
                counts[str(metadata["doc_id"])] += 1
        return dict(counts)

    def get_figure(self, doc_id: str, figure_id: str) -> dict[str, Any] | None:
        result = self.figure_collection.get(ids=[f"{doc_id}:{figure_id}"], include=["documents", "metadatas"])
        documents = result.get("documents") or []
        if not documents:
            return None
        return self._figure_record(documents[0], (result.get("metadatas") or [{}])[0], None)

    def delete_figures(self, doc_id: str) -> None:
        self.figure_collection.delete(where={"doc_id": doc_id})

    def list_documents(self) -> list[dict[str, Any]]:
        result = self.collection.get(include=["metadatas"])
        documents: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"id": "", "filename": "", "ingested_at": "", "chunks": 0}
        )

        for metadata in result.get("metadatas", []):
            if not metadata:
                continue
            doc_id = str(metadata.get("doc_id", ""))
            if not doc_id:
                continue
            record = documents[doc_id]
            record["id"] = doc_id
            record["filename"] = metadata.get("filename", "")
            record["ingested_at"] = metadata.get("ingested_at", "")
            record["chunks"] += 1

        return sorted(documents.values(), key=lambda item: str(item["ingested_at"]), reverse=True)


_store: ChromaStore | None = None


def get_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store
