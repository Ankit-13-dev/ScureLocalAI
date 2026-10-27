import asyncio
import json

import httpx
import pytest

from backend.config import FigureSettings, LlmSettings, Settings, EmbeddingSettings, RetrievalSettings, ServerSettings, StorageSettings
from backend.llm.vision import VisionDescriber
from backend.retrieval.figure_search import (
    build_image_reply,
    analyze_question,
    find_figures,
    keyword_overlap,
)


class FakeStore:
    """Just enough of ChromaStore for ranking tests."""

    def __init__(self, records):
        self.records = records

    def query_figures(self, embedding, k):
        return [dict(r) for r in self.records][:k]

    def figures_by_number(self, number):
        return [dict(r) for r in self.records if r.get("fig_number") == number]

    def list_figures(self, doc_id=None, limit=None):
        return [dict(r) for r in self.records][:limit]


def record(figure_id, caption, distance, number=None, kind="chart"):
    return {
        "doc_id": "d1", "figure_id": figure_id, "filename": "paper.pdf", "page": 2, "kind": kind,
        "caption": caption, "text": f"{caption}", "fig_number": number, "distance": distance,
        "crop_file": f"d1/crops/{figure_id}.png", "bbox": [],
    }


@pytest.mark.parametrize(
    "question, wants_images, wants_text",
    [
        ("show me the workflow diagram", True, False),
        ("Can you provide the image of the face area graph", True, False),
        ("give me Fig. 3", True, False),
        ("What does Fig. 3 show?", False, True),
        ("explain the SSIM graph", False, True),
        ("show me the graph and explain it", True, True),
        ("What is the capital of France?", False, True),
        ("how does the filter work", False, True),
    ],
)
def test_intent_detection(question, wants_images, wants_text) -> None:
    intent = analyze_question(question)
    assert intent.wants_images is wants_images
    assert intent.wants_text is wants_text


def test_figure_numbers_are_extracted() -> None:
    assert analyze_question("show me figures 2 and 4").figure_numbers == (2, 4)
    assert analyze_question("what does Fig. 7 mean").figure_numbers == (7,)


def test_unrelated_request_returns_nothing_instead_of_least_bad_match() -> None:
    store = FakeStore([record("p001_f01", "Fig. 1. High-level workflow of the filter", 0.55)])
    intent = analyze_question("show me a picture of a bird")

    assert find_figures("show me a picture of a bird", intent, [0.0], store, FigureSettings()) == []


def test_keyword_match_wins_even_with_a_loose_embedding_distance() -> None:
    store = FakeStore([
        record("p001_f01", "Fig. 1. High-level workflow of the filter", 0.55),
        record("p002_f01", "Fig. 2. Accuracy per class", 0.50),
    ])
    q = "show me the workflow diagram"
    found = find_figures(q, analyze_question(q), [0.0], store, FigureSettings())

    assert [r["figure_id"] for r in found] == ["p001_f01"]


def test_missing_figure_number_is_a_miss_not_a_generic_answer() -> None:
    store = FakeStore([record("p001_f01", "Fig. 1. Something", 0.3, number=1)])
    q = "give me figure 9"

    assert find_figures(q, analyze_question(q), [0.0], store, FigureSettings()) == []


def test_generic_request_returns_first_images() -> None:
    store = FakeStore([record("p001_f01", "A", 0.9), record("p001_f02", "B", 0.9)])
    q = "show me an image"
    found = find_figures(q, analyze_question(q), [0.0], store, FigureSettings(max_images=1))

    assert len(found) == 1 and found[0]["match"] == "generic"


def test_normal_question_attaches_picture_only_when_confident() -> None:
    store = FakeStore([record("p001_f01", "Fig. 1. SSIM threshold graph", 0.30)])
    settings = FigureSettings()

    q = "explain the SSIM graph"
    assert len(find_figures(q, analyze_question(q), [0.0], store, settings)) == 1
    q = "what is the capital of France"
    assert find_figures(q, analyze_question(q), [0.0], store, settings) == []


def test_reply_text_mentions_caption_page_and_sources() -> None:
    text = build_image_reply([record("p001_f01", "Fig. 1. Workflow", 0.1)])
    assert "Fig. 1. Workflow" in text and "page 2" in text and text.endswith("- paper.pdf")


def test_keyword_overlap_ignores_plurals() -> None:
    assert keyword_overlap(frozenset({"curve"}), "Training curves") == 1.0


# ---------------------------------------------------------------- vision model client
def _settings(mode: str) -> Settings:
    return Settings(
        llm=LlmSettings("qwen3:4b", "http://localhost:11434", 0.1, 128),
        embeddings=EmbeddingSettings("nomic-embed-text", "http://localhost:11434"),
        retrieval=RetrievalSettings(512, 64, 5),
        storage=StorageSettings("a", "b", "test"),
        server=ServerSettings("127.0.0.1", 8000),
        figures=FigureSettings(describe_mode=mode),
    )


def _patch_httpx(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(
        "backend.llm.vision.httpx.AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )


def test_vision_describer_sends_image_and_cleans_reply(monkeypatch) -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "<think>x</think>A bar chart of sales."}})

    _patch_httpx(monkeypatch, handler)
    describer = VisionDescriber(_settings("all"))

    assert asyncio.run(describer.describe(b"png-bytes")) == "A bar chart of sales."
    assert seen["body"]["model"] == "qwen2.5vl:3b"
    assert seen["body"]["messages"][0]["images"]


def test_vision_describer_disables_itself_when_model_is_missing(monkeypatch) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, json={"error": "model not found"})

    _patch_httpx(monkeypatch, handler)
    describer = VisionDescriber(_settings("all"))

    assert asyncio.run(describer.describe(b"x")) is None
    assert asyncio.run(describer.describe(b"x")) is None
    assert len(calls) == 1  # no repeated slow failures


def test_describe_mode_off_and_uncaptioned() -> None:
    assert VisionDescriber(_settings("off")).wants("") is False
    unc = VisionDescriber(_settings("uncaptioned"))
    assert unc.wants("") is True and unc.wants("Fig. 1 something") is False
