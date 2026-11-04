from pathlib import Path

from fastapi.testclient import TestClient

from backend.config import (
    EmbeddingSettings,
    LlmSettings,
    RetrievalSettings,
    ServerSettings,
    Settings,
    StorageSettings,
    VisionSettings,
)
from backend.main import app
from backend.config import get_settings
from backend.retrieval.store import get_store


class FakeStore:
    def query(self, embedding, k):
        return []

    def query_figures(self, embedding, k):
        return []


def test_multimodal_endpoint_forwards_uploaded_image(monkeypatch, tmp_path: Path) -> None:
    import asyncio

    settings = Settings(
        llm=LlmSettings("qwen3:4b", "http://localhost:11434", 0.1, 128),
        embeddings=EmbeddingSettings("nomic-embed-text", "http://localhost:11434"),
        retrieval=RetrievalSettings(512, 64, 5),
        storage=StorageSettings(tmp_path / "documents", tmp_path / "chroma", "test"),
        server=ServerSettings("127.0.0.1", 8000),
        vision=VisionSettings("qwen2.5vl:3b", "http://localhost:11434", True, 100),
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_store] = lambda: FakeStore()

    async def fake_embed(texts, embed_settings):
        return [[0.1, 0.2] for _ in texts]

    captured = {}

    async def fake_answer(question, context, llm_settings, image_bytes=None, vision_settings=None):
        captured["question"] = question
        captured["images"] = image_bytes
        captured["vision_model"] = vision_settings.model
        return "This image contains a test diagram."

    monkeypatch.setattr("backend.routers.query.embed", fake_embed)
    monkeypatch.setattr("backend.routers.query.answer_query", fake_answer)

    try:
        client = TestClient(app)
        response = client.post(
            "/api/query/multimodal",
            data={"question": "Explain this diagram"},
            files={"image": ("diagram.png", b"fake-png-bytes", "image/png")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "This image contains a test diagram."
    assert captured["question"] == "Explain this diagram"
    assert captured["images"] == [b"fake-png-bytes"]
    assert captured["vision_model"] == "qwen2.5vl:3b"
