from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from backend.config import Settings, get_settings
from backend.retrieval.figure_search import image_url
from backend.retrieval.store import ChromaStore, get_store


router = APIRouter(prefix="/figures", tags=["figures"])

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


@router.get("")
async def list_figures(doc_id: str | None = None, store: ChromaStore = Depends(get_store)) -> list[dict[str, object]]:
    """Every indexed picture (optionally for one document), with the URL of its image."""
    records = store.list_figures(doc_id)
    return [
        {
            "doc_id": r["doc_id"],
            "figure_id": r["figure_id"],
            "filename": r["filename"],
            "page": r["page"],
            "kind": r["kind"],
            "caption": r["caption"] or None,
            "url": image_url(r),
        }
        for r in records
    ]


@router.get("/{doc_id}/{figure_id}/image")
async def get_figure_image(
    doc_id: str,
    figure_id: str,
    settings: Settings = Depends(get_settings),
    store: ChromaStore = Depends(get_store),
) -> FileResponse:
    if not _SAFE_ID.match(doc_id) or not _SAFE_ID.match(figure_id):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid figure id")

    record = store.get_figure(doc_id, figure_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    root = settings.figures_root.resolve()
    path = (root / record["crop_file"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():  # blocks path traversal, missing files
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image file is missing")

    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})
