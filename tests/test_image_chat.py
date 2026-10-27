"""End-to-end: upload a PDF -> pictures are indexed -> chat returns the right image file."""

import hashlib
import math
import re
import shutil
from pathlib import Path

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("fastapi")
pytest.importorskip("pymupdf")
pytest.importorskip("cv2")

from fastapi.testclient import TestClient  # noqa: E402

from backend.config import (  # noqa: E402
    EmbeddingSettings,
    LlmSettings,
    RetrievalSettings,
    ServerSettings,
    Settings,
    StorageSettings,
    get_settings,
)
from backend.ingestion.Visualizing import build_selftest_pdf  # noqa: E402
from backend.main import app  # noqa: E402
from backend.retrieval.store import ChromaStore, get_store  # noqa: E402


def hash_embedding(text: str, dims: int = 256) -> list[float]:
    """Stand-in for nomic-embed-text: hashed bag-of-words, so similar wording -> similar vector."""
    vec = [0.0] * dims
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture()
def client(tmp_path: Path, monkeypatch):
    settings = Settings(
        llm=LlmSettings("qwen3:4b", "http://localhost:11434", 0.1, 128),
        embeddings=EmbeddingSettings("nomic-embed-text", "http://localhost:11434"),
        retrieval=RetrievalSettings(512, 64, 5),
        storage=StorageSettings(tmp_path / "documents", tmp_path / "chroma", "test", tmp_path / "figures"),
        server=ServerSettings("127.0.0.1", 8000),
    )
    store = ChromaStore(settings)

    async def fake_embed(texts, _settings=None):
        return [hash_embedding(t) for t in texts]

    calls: list[str] = []

    async def fake_answer(question, chunks, _settings=None):
        calls.append(question)
        return "LLM ANSWER"

    monkeypatch.setattr("backend.ingestion.pipeline.embed", fake_embed)
    monkeypatch.setattr("backend.ingestion.figures.embed", fake_embed)
    monkeypatch.setattr("backend.routers.query.embed", fake_embed)
    monkeypatch.setattr("backend.routers.query.answer_query", fake_answer)

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_store] = lambda: store
    test_client = TestClient(app)
    test_client.llm_calls = calls  # type: ignore[attr-defined]
    test_client.settings = settings  # type: ignore[attr-defined]
    test_client.store = store  # type: ignore[attr-defined]
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def demo_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "demo paper.pdf"
    build_selftest_pdf(path)
    return path


def upload(client: TestClient, path: Path) -> dict:
    response = client.post("/api/documents", files={"file": (path.name, path.read_bytes(), "application/pdf")})
    assert response.status_code == 201, response.text
    return response.json()


def ask(client: TestClient, question: str) -> dict:
    response = client.post("/api/query", json={"question": question})
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_indexes_pictures_and_lists_them(client, demo_pdf) -> None:
    result = upload(client, demo_pdf)

    assert result["chunks"] > 0
    assert result["figures"] == 7
    documents = client.get("/api/documents").json()
    assert documents[0]["figures"] == 7
    assert len(client.get("/api/figures").json()) == 7


def test_show_me_a_figure_by_number_returns_the_image_file(client, demo_pdf) -> None:
    upload(client, demo_pdf)

    body = ask(client, "Show me Figure 3")

    assert len(body["images"]) == 1
    image = body["images"][0]
    assert image["page"] == 2 and image["kind"] == "photo"
    assert "landscape photograph" in body["answer"]
    assert client.llm_calls == []  # the picture is the answer; the LLM was not involved

    png = client.get(image["url"])
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    assert png.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_show_me_by_description(client, demo_pdf) -> None:
    upload(client, demo_pdf)

    body = ask(client, "can you provide me the photograph of the landscape")

    assert body["images"] and body["images"][0]["kind"] == "photo"
    assert client.llm_calls == []


def test_request_for_something_that_is_not_indexed_is_refused_honestly(client, demo_pdf) -> None:
    upload(client, demo_pdf)

    body = ask(client, "show me a picture of a bird")

    assert body["images"] == []
    assert "could not find a matching image" in body["answer"]
    assert "cannot create new images" in body["answer"]


def test_question_about_a_figure_gets_llm_answer_plus_the_picture(client, demo_pdf) -> None:
    upload(client, demo_pdf)

    body = ask(client, "What does Figure 3 show?")

    assert body["answer"].startswith("LLM ANSWER")
    assert [i["page"] for i in body["images"]] == [2]


def test_plain_text_questions_get_no_images(client, demo_pdf) -> None:
    upload(client, demo_pdf)

    body = ask(client, "What is the capital of France?")

    assert body["images"] == []
    assert body["answer"] == "LLM ANSWER"


def test_image_endpoint_rejects_bad_and_unknown_ids(client, demo_pdf) -> None:
    doc = upload(client, demo_pdf)

    assert client.get("/api/figures/not-a-doc/p001_f01/image").status_code == 404
    assert client.get(f"/api/figures/{doc['id']}/..%2F..%2Fsecret/image").status_code in (400, 404)
    assert client.get(f"/api/figures/{doc['id']}/p999_f99/image").status_code == 404


def test_delete_removes_images_from_index_and_disk(client, demo_pdf) -> None:
    doc = upload(client, demo_pdf)
    url = ask(client, "Show me Figure 3")["images"][0]["url"]
    figures_dir = client.settings.figures_root / doc["id"]
    assert figures_dir.exists()

    assert client.delete(f"/api/documents/{doc['id']}").status_code == 204

    assert not figures_dir.exists()
    assert client.get(url).status_code == 404
    assert client.get("/api/figures").json() == []
    assert "No images have been indexed" in ask(client, "show me the diagram")["answer"]


def test_rescan_restores_pictures_of_documents_uploaded_earlier(client, demo_pdf) -> None:
    doc = upload(client, demo_pdf)
    client.store.delete_figures(doc["id"])
    shutil.rmtree(client.settings.figures_root / doc["id"])
    assert client.get("/api/figures").json() == []

    report = client.post("/api/documents/rescan-images").json()

    assert report["images"] == 7
    assert len(client.get("/api/figures").json()) == 7


def test_text_ingestion_survives_a_failing_image_step(client, demo_pdf, monkeypatch) -> None:
    async def boom(*args, **kwargs):
        raise RuntimeError("vision stack exploded")

    monkeypatch.setattr("backend.ingestion.pipeline.ingest_figures", boom)

    result = upload(client, demo_pdf)

    assert result["chunks"] > 0 and result["figures"] == 0


def test_text_only_files_are_unaffected(client, tmp_path) -> None:
    note = tmp_path / "note.txt"
    note.write_text("Alpha beta gamma.", encoding="utf-8")

    response = client.post("/api/documents", files={"file": ("note.txt", note.read_bytes(), "text/plain")})

    assert response.status_code == 201
    assert response.json()["figures"] == 0
    assert ask(client, "What is alpha?")["images"] == []
