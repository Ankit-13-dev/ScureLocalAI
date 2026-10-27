from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


@dataclass(frozen=True)
class LlmSettings:
    model: str
    base_url: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class EmbeddingSettings:
    model: str
    base_url: str


@dataclass(frozen=True)
class RetrievalSettings:
    chunk_size: int
    chunk_overlap: int
    top_k: int


@dataclass(frozen=True)
class StorageSettings:
    documents_dir: Path
    chroma_dir: Path
    chroma_collection: str
    figures_dir: Path | None = None  # where cropped images live; defaults to <data>/figures


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int


@dataclass(frozen=True)
class FigureSettings:
    """Image retrieval: detect pictures in PDFs, index them, and show them in chat."""

    enabled: bool = True
    describe_mode: str = "off"            # off | uncaptioned | all  (local vision model descriptions)
    vlm_model: str = "qwen2.5vl:3b"       # any Ollama vision model
    vlm_base_url: str = ""                # empty -> reuse llm.base_url
    vlm_timeout_s: float = 240.0
    max_images: int = 3                   # images returned per answer
    semantic_max_distance: float = 0.40   # no shared keywords: accept a match only this close (cosine distance)
    min_keyword_overlap: float = 0.34     # share of the question's keywords found in the figure text


@dataclass(frozen=True)
class Settings:
    llm: LlmSettings
    embeddings: EmbeddingSettings
    retrieval: RetrievalSettings
    storage: StorageSettings
    server: ServerSettings
    figures: FigureSettings = field(default_factory=FigureSettings)

    @property
    def figures_root(self) -> Path:
        return self.storage.figures_dir or (self.storage.chroma_dir.parent / "figures")


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def _read_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a mapping at the top level")
    return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data = _read_config()
    llm = data.get("llm", {})
    embeddings = data.get("embeddings", {})
    retrieval = data.get("retrieval", {})
    storage = data.get("storage", {})
    server = data.get("server", {})
    figures = data.get("figures", {}) or {}
    figure_defaults = FigureSettings()

    return Settings(
        llm=LlmSettings(
            model=str(llm["model"]),
            base_url=str(llm["base_url"]).rstrip("/"),
            temperature=float(llm.get("temperature", 0.1)),
            max_tokens=int(llm.get("max_tokens", 2048)),
        ),
        embeddings=EmbeddingSettings(
            model=str(embeddings["model"]),
            base_url=str(embeddings["base_url"]).rstrip("/"),
        ),
        retrieval=RetrievalSettings(
            chunk_size=int(retrieval.get("chunk_size", 512)),
            chunk_overlap=int(retrieval.get("chunk_overlap", 64)),
            top_k=int(retrieval.get("top_k", 5)),
        ),
        storage=StorageSettings(
            documents_dir=_project_path(str(storage.get("documents_dir", "data/documents"))),
            chroma_dir=_project_path(str(storage.get("chroma_dir", "data/chroma_db"))),
            chroma_collection=str(storage.get("chroma_collection", "scurelocalai")),
            figures_dir=_project_path(str(storage.get("figures_dir", "data/figures"))),
        ),
        server=ServerSettings(
            host=str(server.get("host", "127.0.0.1")),
            port=int(server.get("port", 8000)),
        ),
        figures=FigureSettings(
            enabled=bool(figures.get("enabled", figure_defaults.enabled)),
            describe_mode=str(figures.get("describe_mode", figure_defaults.describe_mode)).lower(),
            vlm_model=str(figures.get("vlm_model", figure_defaults.vlm_model)),
            vlm_base_url=str(figures.get("vlm_base_url", figure_defaults.vlm_base_url)).rstrip("/"),
            vlm_timeout_s=float(figures.get("vlm_timeout_s", figure_defaults.vlm_timeout_s)),
            max_images=int(figures.get("max_images", figure_defaults.max_images)),
            semantic_max_distance=float(figures.get("semantic_max_distance", figure_defaults.semantic_max_distance)),
            min_keyword_overlap=float(figures.get("min_keyword_overlap", figure_defaults.min_keyword_overlap)),
        ),
    )


def ensure_storage_dirs(settings: Settings | None = None) -> None:
    resolved = settings or get_settings()
    resolved.storage.documents_dir.mkdir(parents=True, exist_ok=True)
    resolved.storage.chroma_dir.mkdir(parents=True, exist_ok=True)
    resolved.figures_root.mkdir(parents=True, exist_ok=True)
