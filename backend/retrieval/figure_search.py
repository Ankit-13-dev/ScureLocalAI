"""Decide when a question is about a picture, and find the right stored crop.

Everything here is deterministic and offline.  The embedding of the question (already computed
for text retrieval) is compared with the stored picture records, then re-ranked with a keyword
check so that an unrelated picture is never returned just because it is the "least bad" match.

Three situations are distinguished:

1. Explicit request   "show me the workflow diagram", "give me Fig. 3", "send the pie chart"
                      -> the answer *is* the picture(s); the LLM is skipped (nothing to hallucinate).
2. Figure reference   "what does Fig. 3 show?", "explain the SSIM graph"
                      -> normal LLM answer, plus the matching picture attached.
3. Everything else    -> unchanged text-only answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from backend.config import FigureSettings
from backend.retrieval.store import ChromaStore


VISUAL_NOUN = re.compile(
    r"\b(images?|pictures?|pics?|photos?|photographs?|figures?|figs?|diagrams?|graphs?|charts?|plots?|"
    r"illustrations?|flow\s?charts?|screenshots?|visuals?|drawings?|sketch(?:es)?|infographics?|schematics?)\b",
    re.IGNORECASE,
)
REQUEST_VERB = re.compile(
    r"\b(show|display|give|provide|send|share|fetch|find|get|see|view|open|bring|attach|retrieve|"
    r"present|print|generate|produce|draw|look\s+at|pull\s+up)\b",
    re.IGNORECASE,
)
TEXT_WORDS = re.compile(
    r"\b(explain|describe|what|why|how|summari[sz]e|interpret|mean|means|meaning|tell|analy[sz]e|compare|discuss)\b",
    re.IGNORECASE,
)
FIG_NUMBERS = re.compile(r"\bfig(?:ure)?s?\.?\s*#?((?:\d{1,3}\s*(?:,|and|&)?\s*)+)", re.IGNORECASE)

# words that carry no information about *which* picture is wanted
STOPWORDS = frozenset(
    """a an the of in on at to for from with and or but is are was were be been being this that these those
    it its me my you your we our can could would should will please any some all there here about into out
    do does did have has had give show send find get see view open bring provide display fetch share attach
    retrieve present print generate produce draw look pull want need like let which where when who whom
    image images picture pictures pic pics figure figures fig figs visual visuals one ones also just then
    than them they their his her him she he not no yes too very more most much many what why how tell
    explain describe summarize summarise page pages document documents file files pdf""".split()
)

IMAGE_NOT_FOUND_MESSAGE = (
    "I could not find a matching image in the knowledge base. I can only show pictures that appear in "
    "your uploaded documents; I cannot create new images."
)
NO_IMAGES_INDEXED_MESSAGE = (
    "No images have been indexed yet. Upload a PDF (pictures are indexed automatically), or use "
    "\"Scan images\" to index the pictures in documents you uploaded earlier."
)


@dataclass(frozen=True)
class QueryIntent:
    wants_images: bool
    wants_text: bool
    mentions_visual: bool
    figure_numbers: tuple[int, ...]
    keywords: frozenset[str]


def _normalise(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {_normalise(t) for t in tokens if len(t) >= 3 and t not in STOPWORDS and not t.isdigit()}


def analyze_question(question: str) -> QueryIntent:
    nouns = [m.start() for m in VISUAL_NOUN.finditer(question)]
    verbs = [m.start() for m in REQUEST_VERB.finditer(question)]
    # "show me the graph" is a request (verb before the noun); "what does Fig. 3 show" is not.
    verb_before_noun = any(v < n for v in verbs for n in nouns)
    noun_first = bool(re.match(r"\s*(?:the\s+|a\s+|an\s+)?(images?|pictures?|photos?|diagrams?|graphs?|charts?|plots?)\b", question, re.I))

    numbers: list[int] = []
    for match in FIG_NUMBERS.finditer(question):
        numbers.extend(int(n) for n in re.findall(r"\d{1,3}", match.group(1)))

    return QueryIntent(
        wants_images=bool(nouns) and (verb_before_noun or noun_first),
        wants_text=bool(TEXT_WORDS.search(question)),
        mentions_visual=bool(nouns),
        figure_numbers=tuple(dict.fromkeys(numbers)),
        keywords=frozenset(tokenize(question)),
    )


def keyword_overlap(keywords: frozenset[str], text: str) -> float:
    if not keywords:
        return 0.0
    return len(keywords & tokenize(text)) / len(keywords)


def _score(distance: float | None, overlap: float) -> float:
    similarity = 0.0 if distance is None else max(0.0, 1.0 - distance)
    return 0.65 * similarity + 0.35 * overlap


def find_figures(
    question: str,
    intent: QueryIntent,
    embedding: list[float] | None,
    store: ChromaStore,
    settings: FigureSettings,
) -> list[dict[str, Any]]:
    """Return the pictures to show for this question (possibly none), best first."""
    limit = max(1, settings.max_images)
    semantic: list[dict[str, Any]] = []
    if embedding is not None and (intent.wants_images or intent.mentions_visual):
        semantic = store.query_figures(embedding, 10)
    for record in semantic:
        record["overlap"] = keyword_overlap(intent.keywords, record["text"])
        record["score"] = _score(record["distance"], record["overlap"])

    # 1. "Fig. 3" is exact: no similarity threshold needed.
    if intent.figure_numbers:
        numbered: dict[str, dict[str, Any]] = {}
        for number in intent.figure_numbers:
            for record in store.figures_by_number(number):
                numbered[f"{record['doc_id']}:{record['figure_id']}"] = record
        if numbered:
            known = {f"{r['doc_id']}:{r['figure_id']}": r for r in semantic}
            for key, record in numbered.items():
                record.update({k: known[key][k] for k in ("distance", "overlap", "score") if key in known})
                record.setdefault("overlap", keyword_overlap(intent.keywords, record["text"]))
                record.setdefault("score", _score(None, record["overlap"]))
                record["match"] = "number"
            ordered = sorted(numbered.values(), key=lambda r: (-r["overlap"], str(r["filename"]), r["page"] or 0))
            return ordered[:limit]

    if not (intent.wants_images or intent.mentions_visual):
        return []

    # 2. "show me an image" with nothing to match on: hand back the first pictures indexed.
    #    (A figure number that does not exist is a miss, not a generic request.)
    if intent.wants_images and not intent.keywords and intent.figure_numbers:
        return []
    if intent.wants_images and not intent.keywords:
        generic = store.list_figures(limit=limit)
        for record in generic:
            record["match"] = "generic"
        return generic

    def accepted(record: dict[str, Any]) -> bool:
        # Evidence of relevance: the question's own words appear in the figure's caption/text,
        # or the meaning is very close.  "Least bad" matches are never returned.
        if record["overlap"] >= settings.min_keyword_overlap:
            return True
        return record["distance"] is not None and record["distance"] <= settings.semantic_max_distance

    ranked = sorted(semantic, key=lambda r: -r["score"])
    if intent.wants_images:
        good = [r for r in ranked if accepted(r)]
        if not good:
            return []
        top = good[0]["score"]
        chosen = [r for r in good if r["score"] >= 0.85 * top][:limit]
    else:
        # A normal question that merely mentions a picture: attach only a confident single match.
        top1 = ranked[0] if ranked else None
        if top1 is None or not accepted(top1):
            return []
        chosen = [top1]
    for record in chosen:
        record["match"] = "semantic"
    return chosen


# ---------------------------------------------------------------------------- presentation
def image_url(record: dict[str, Any]) -> str:
    return f"/api/figures/{record['doc_id']}/{record['figure_id']}/image"


def to_image_ref(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"{record['doc_id']}:{record['figure_id']}",
        "doc_id": record["doc_id"],
        "figure_id": record["figure_id"],
        "filename": record["filename"],
        "page": record["page"],
        "kind": record["kind"],
        "caption": record["caption"] or None,
        "url": image_url(record),
        "match": record.get("match", "semantic"),
        "score": round(float(record.get("score", 0.0)), 3),
    }


def _label(record: dict[str, Any]) -> str:
    caption = (record.get("caption") or "").strip()
    if caption:
        return caption if len(caption) <= 200 else caption[:200].rstrip() + "..."
    kind = {"photo": "photograph", "chart": "chart", "diagram_or_illustration": "diagram"}.get(record.get("kind"), "image")
    return f"a {kind}"


def build_image_reply(records: list[dict[str, Any]]) -> str:
    if len(records) == 1:
        r = records[0]
        body = f"Here is the image I found: {_label(r)} ({r['filename']}, page {r['page']})."
    else:
        lines = [f"- {_label(r)} ({r['filename']}, page {r['page']})" for r in records]
        body = "Here are the closest images I found:\n" + "\n".join(lines)
    sources: list[str] = []
    for r in records:
        if r["filename"] and r["filename"] not in sources:
            sources.append(r["filename"])
    return body + "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)


def figure_context_chunks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Let the LLM read what is known about attached pictures (caption, description, citations)."""
    return [
        {"text": f"[Image on this page] {r['text']}", "filename": r["filename"], "page": r["page"]}
        for r in records
    ]
