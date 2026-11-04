from backend.config import LlmSettings
from backend.llm.chat import NOT_FOUND_MESSAGE, answer_query, ensure_sources


def test_answer_query_short_circuits_empty_context() -> None:
    import asyncio

    answer = asyncio.run(answer_query("What happened?", []))

    assert answer == NOT_FOUND_MESSAGE


def test_ensure_sources_appends_missing_sources() -> None:
    answer = ensure_sources("The answer is grounded.", [{"filename": "source.txt"}])

    assert answer.endswith("Sources:\n- source.txt")


def test_answer_query_strips_thinking_and_enforces_sources(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"message": {"content": "<think>private</think>The answer."}}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def post(self, *args, **kwargs) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr("backend.llm.chat.httpx.AsyncClient", FakeAsyncClient)

    import asyncio

    answer = asyncio.run(
        answer_query(
            "Question?",
            [{"text": "Context", "filename": "doc.txt", "page": 1}],
            LlmSettings(
                model="qwen3:4b",
                base_url="http://localhost:11434",
                temperature=0.1,
                max_tokens=128,
            ),
        )
    )

    assert "<think>" not in answer
    assert answer == "The answer.\n\nSources:\n- doc.txt"


def test_simple_mode_uses_the_tutor_prompt_and_normal_mode_does_not(monkeypatch) -> None:
    import asyncio

    from backend.llm.chat import RAG_SYSTEM_PROMPT, SIMPLE_SYSTEM_PROMPT, _pick_system_prompt

    assert _pick_system_prompt("simple") == SIMPLE_SYSTEM_PROMPT
    assert _pick_system_prompt("normal") == RAG_SYSTEM_PROMPT
    assert _pick_system_prompt("anything-else") == RAG_SYSTEM_PROMPT   # unknown values fall back safely

    sent: list[str] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"message": {"content": "ok"}}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def post(self, url, json=None, **kwargs):
            sent.append(json["messages"][0]["content"])
            return FakeResponse()

    monkeypatch.setattr("backend.llm.chat.httpx.AsyncClient", FakeAsyncClient)
    chunks = [{"text": "Context", "filename": "doc.txt", "page": 1}]
    llm = LlmSettings(model="qwen3:4b", base_url="http://localhost:11434", temperature=0.0, max_tokens=64)

    asyncio.run(answer_query("Q?", chunks, llm))
    asyncio.run(answer_query("Q?", chunks, llm, mode="simple"))

    assert sent[0].startswith(RAG_SYSTEM_PROMPT) and "friendly tutor" not in sent[0]
    assert sent[1].startswith(SIMPLE_SYSTEM_PROMPT) and "friendly tutor" in sent[1]
