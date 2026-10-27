from __future__ import annotations

import logging
import re
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.config import Settings, ensure_storage_dirs, get_settings
from backend.ingestion.figures import ingest_figures
from backend.ingestion.pipeline import PARSERS, ingest_document
from backend.retrieval.store import ChromaStore, get_store


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = set(PARSERS.keys())
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    return name or "document"


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    store: ChromaStore = Depends(get_store),
) -> dict[str, object]:
    filename = _safe_filename(file.filename or "")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    ensure_storage_dirs(settings)
    doc_id = str(uuid.uuid4())
    destination = settings.storage.documents_dir / f"{doc_id}_{filename}"

    try:
        content = await file.read()
        destination.write_bytes(content)
        chunks = await ingest_document(destination, doc_id, filename, settings=settings, store=store)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to ingest uploaded document")
        if destination.exists():
            destination.unlink()
        shutil.rmtree(settings.figures_root / doc_id, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {exc}",
        ) from exc
    finally:
        await file.close()

    return {
        "id": doc_id,
        "filename": filename,
        "chunks": chunks,
        "figures": store.figure_counts().get(doc_id, 0),
    }


@router.get("")
async def list_documents(store: ChromaStore = Depends(get_store)) -> list[dict[str, object]]:
    documents = store.list_documents()
    figure_counts = store.figure_counts()
    for document in documents:
        document["figures"] = figure_counts.get(str(document["id"]), 0)
    return documents


@router.post("/rescan-images")
async def rescan_images(
    settings: Settings = Depends(get_settings),
    store: ChromaStore = Depends(get_store),
) -> dict[str, object]:
    """Index the pictures of PDFs that were uploaded before image search existed (idempotent)."""
    scanned: list[dict[str, object]] = []
    for document in store.list_documents():
        doc_id, filename = str(document["id"]), str(document["filename"])
        if not filename.lower().endswith(".pdf") or not _SAFE_ID.match(doc_id):
            continue
        path = settings.storage.documents_dir / f"{doc_id}_{filename}"
        if not path.is_file():
            scanned.append({"filename": filename, "images": 0, "error": "original file not found"})
            continue
        try:
            count = await ingest_figures(path, doc_id, filename, settings=settings, store=store)
            scanned.append({"filename": filename, "images": count})
        except Exception as exc:  # noqa: BLE001 - keep scanning the other documents
            logger.exception("Image scan failed for %s", filename)
            scanned.append({"filename": filename, "images": 0, "error": str(exc)})
    return {"documents": scanned, "images": sum(int(item["images"]) for item in scanned)}


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: str,
    settings: Settings = Depends(get_settings),
    store: ChromaStore = Depends(get_store),
) -> None:
    if not _SAFE_ID.match(doc_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document id")
    store.delete_document(doc_id)  # text chunks + image records
    shutil.rmtree(settings.figures_root / doc_id, ignore_errors=True)  # cropped images on disk
    for path in settings.storage.documents_dir.glob(f"{doc_id}_*"):
        if path.is_file():
            path.unlink()
