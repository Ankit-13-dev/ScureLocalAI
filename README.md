# ScureLocalAI

ScureLocalAI is an offline, local-first personal knowledge base with document upload, local vector search, and an Ollama-powered RAG chat interface.

Upload documents, index them locally, and ask grounded questions against your own files. The app uses local embeddings, a local ChromaDB vector store, and a local LLM served by Ollama.

No cloud LLM APIs are used. No API keys are required. Uploaded documents, embeddings, and vector indexes stay on your machine.


---

## Project Status

**Early alpha / local prototype.** The core offline upload, retrieval, and chat flow works, but the project is still being hardened. Expect rough edges around installation environments, parser quality, scanned PDFs, and large-document performance.

This repository is public so others can inspect, run, fork, and build on the local-first RAG approach.

---

## What Works Today

- Upload and index `.pdf`, `.docx`, `.pptx`, `.txt`, and `.md` files
- Store uploaded files and vector data locally under `data/`
- Generate embeddings locally through Ollama
- Retrieve relevant chunks with ChromaDB
- Answer questions with a local Ollama chat model
- Return source filenames with answers
- Find pictures, charts, graphs and diagrams in uploaded PDFs and show them in chat on request ("show me the workflow diagram", "show me Fig. 3")
- Refuse answers when retrieved context is insufficient
- Run the frontend without a JavaScript build step

---

## Why This Exists

Many RAG-style knowledge-base tools rely on hosted LLMs, hosted embedding APIs, telemetry, or managed vector databases. ScureLocalAI is intentionally small and local:

- FastAPI backend
- Vanilla HTML/CSS/JS frontend
- Ollama for chat and embeddings
- ChromaDB for local vector storage
- Plain file upload workflow
- No cloud API dependency at runtime

---

## Privacy And Offline Model

After the one-time dependency/model setup:

- The app runs on `127.0.0.1`
- Documents are stored under `data/documents/`
- Vector data is stored under `data/chroma_db/`
- Ollama serves models locally through `localhost:11434`
- No OpenAI, Anthropic, Gemini, Cohere, Pinecone, Weaviate Cloud, or other cloud SDK is used

Model downloads are handled by Ollama during setup. Model weights are not stored in this repository.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI, Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Chat LLM | `qwen3:4b` via Ollama |
| Embeddings | `nomic-embed-text` via Ollama |
| Vector store | ChromaDB persistent local store |
| Parsers | pypdf, python-docx, python-pptx, built-in text parsing |
| Tests | pytest, pytest-asyncio |

---

## Quick Start

### Prerequisites

- Python 3.11 recommended
- [Ollama](https://ollama.com/download)
- About 4 GB free disk space for the default models

### Clone

```bash
git clone <your-repo-url>
cd ScureLocalAI
```

### Setup

macOS / Linux:

```bash
bash setup.sh
```

Windows PowerShell:

```powershell
.\setup.bat
```

The setup script installs Python dependencies and pulls:

- `qwen3:4b`
- `nomic-embed-text`

Optional but recommended (image descriptions): `ollama pull qwen2.5vl:3b`. Optional (scanned PDFs): install the Tesseract program, see [docs/setup.md](docs/setup.md).

### Run

In one terminal:

```bash
ollama serve
```

In another terminal:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

---

## Hardware Requirements

| Configuration | RAM | Disk |
|---|---:|---:|
| Default (`qwen3:4b`) | 8 GB | ~3 GB for models |
| Higher quality (`qwen3:8b`) | 16 GB | ~5.5 GB for models |
| Low-RAM (`phi3:mini`) | 4 GB | ~2.5 GB for models |

To switch models, edit `config.yaml`:

```yaml
llm:
  model: qwen3:8b
```

Then pull the selected model:

```bash
ollama pull qwen3:8b
```

---

## Usage

### Add Documents

1. Open `http://127.0.0.1:8000`
2. Drag and drop a supported file, or click the upload area
3. Wait for the upload/indexing status to complete
4. Confirm the document appears in the sidebar

Supported formats:

- `.pdf`
- `.docx`
- `.pptx`
- `.txt`
- `.md`

### Ask Questions

Type a question into the chat input and press **Send**.

The assistant will:

1. Embed the question locally
2. Retrieve relevant chunks from ChromaDB
3. Ask the local LLM to answer only from those chunks
4. Show source filenames below the answer

If the retrieved context is insufficient, it should respond:

```text
I could not find relevant information in the knowledge base.
```

### Document Sections And Linking Tools

PDF uploads also record each chunk's and image's chapter/section (`structure:` in `config.yaml`). `python backend/ingestion/structure.py file.pdf` prints the detected outline. To measure image-to-text linking on your own documents, see `tools/README.md`.

### Ask For Images

Pictures inside uploaded PDFs (photos, bar/pie/line charts, diagrams, flowcharts, scanned figures) are detected, cropped and indexed automatically on upload. Then you can ask:

- `show me the workflow diagram`
- `give me Fig. 3`
- `send the graph about face area`
- `what does Fig. 2 show?` (normal answer, with the picture attached)

The chat displays the stored crop with its caption, document and page. The assistant only **retrieves** pictures that exist in your documents; it never generates new ones, and it says so when nothing matches. Documents uploaded before this feature existed can be indexed with the **Scan images** button. Scanned PDFs work through pixel analysis. Images inside `.docx` / `.pptx` files are not indexed yet.

### Image Descriptions

With the vision model installed (`ollama pull qwen2.5vl:3b`), every picture is described on upload (people and clothing colours, counts, objects, visible text, diagram labels). The description is used in two ways:

- it makes pictures findable even without a caption ("the picture with the children in green tops"), and
- it is stored as searchable text, so ordinary questions ("how many children are shown?") can be answered from it - and the picture is shown with the answer.

Settings are under `figures:` in `config.yaml` (`describe_mode: all | uncaptioned | off`, `describe_prompt: detailed | brief`). If the vision model is not installed, descriptions are skipped and everything else keeps working. Describing many pictures makes uploads slower.

### Scanned PDFs (OCR)

Pages with no text layer (scans, phone photos saved as PDF) are read with Tesseract, offline, and indexed like normal text (marked `[Scanned page - OCR]`). This needs the Tesseract program (see [docs/setup.md](docs/setup.md)); without it, scanned pages stay empty as before. Settings: `ocr:` in `config.yaml`.

### Simple Mode

The **Simple** button next to Send switches answers to a plain-language tutor style: short bullet points, everyday words, an analogy, and a quick self-check question. Answers still come only from your documents. Click it again to return to normal precise answers.

---

## API Summary

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/documents` | Upload and index a document |
| `GET` | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents/{id}` | Delete a document, its chunks and its cropped images |
| `POST` | `/api/documents/rescan-images` | Index pictures of PDFs uploaded earlier |
| `POST` | `/api/query` | Ask a question; response includes `answer`, `sources` and `images` |
| `GET` | `/api/figures` | List indexed pictures |
| `GET` | `/api/figures/{doc_id}/{figure_id}/image` | The cropped PNG of one picture |

See [docs/architecture.md](docs/architecture.md) for the data flow.

---

## Anti-Hallucination Behavior

The LLM prompt requires answers to use only retrieved excerpts. The backend also enforces source metadata. This does not make the system perfect, but it gives the app a conservative default behavior:

- Empty context returns a fixed "not found" response
- The model is instructed to answer only from excerpts
- Source filenames are returned separately by the API
- The UI deduplicates displayed source names

---

## Configuration

All runtime settings live in `config.yaml`:

```yaml
llm:
  model: qwen3:4b
  base_url: http://localhost:11434
  temperature: 0.0
  max_tokens: 2048

embeddings:
  model: nomic-embed-text
  base_url: http://localhost:11434

retrieval:
  chunk_size: 600
  chunk_overlap: 100
  top_k: 8

figures:
  enabled: true
  describe_mode: "all"        # off | uncaptioned | all
  describe_prompt: detailed   # detailed | brief
  max_images: 3

ocr:
  enabled: true               # needs the Tesseract program; skipped if missing
  language: eng
```

Detection tuning for pictures lives under `vision:` (see `backend/ingestion/Visualizing.py`).

---

## Development

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Use Python 3.11 for the current secure ChromaDB pin. On Windows, Python 3.12 may try to build a native Chroma dependency from source and require Microsoft C++ Build Tools.

Run tests:

```bash
python -m pytest tests -v
```

The repository includes a GitHub Actions workflow at [.github/workflows/tests.yml](.github/workflows/tests.yml).

### Contributing / Forking

This is an early local-first RAG project. Useful areas to explore:

- Better document parsing and layout preservation
- Better OCR quality (language packs, layout) for scanned PDFs
- More file types, such as CSV, XLSX, HTML, or EPUB
- Improved chunking strategies
- Better relevance filtering and retrieval scoring
- Import/export for indexes
- More robust UI states and accessibility checks
- Packaging for easier local installation

Keep changes aligned with the local-only constraint: no runtime cloud APIs, no telemetry, no model weights in the repo, and no dependencies that require user data to leave the machine.

---

## Project Structure

```text
ScureLocalAI/
|-- backend/        Python FastAPI backend
|-- frontend/       HTML/CSS/JS web interface
|-- data/           Runtime documents and vector DB, gitignored
|-- docs/           Architecture notes and setup guide
|-- tests/          pytest test suite
|-- RUNNING.md      Day-to-day startup guide
|-- config.yaml     Runtime configuration
`-- requirements.txt
```

---

## Known Limitations

- This is an early alpha, not a production-grade document management system.
- OCR for scanned PDFs needs the Tesseract program and works best on clean, straight scans; skewed or low-quality phone photos, handwriting, tables and equations read poorly. Pictures on scanned pages are located by pixel analysis, which is less precise than for digital PDFs.
- Image retrieval depends on captions, nearby text and vision-model descriptions; uncaptioned pictures can be missed if `figures.describe_mode` is `off` or the vision model is not installed. Match thresholds in `config.yaml` may need tuning for your documents.
- Parser quality varies by file type and document layout.
- Very large documents may take time to parse, embed, and store.
- Model quality depends on the selected Ollama model and local hardware.
- The app is designed for single-user local use, not multi-user hosting.

---

## Legal And Attribution Notes

- Project code is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
- Third-party Python dependencies, tools, and model references are documented in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
- Ollama is not bundled with this repository. Users install it separately.
- Model weights are not redistributed in this repository. Users download models through Ollama and are responsible for complying with each model's license.
- Uploaded documents are user-provided content. Users are responsible for ensuring they have the right to process and store the files they upload.

---

## Related Documentation

- [Running the app](RUNNING.md)
- [Setup guide](docs/setup.md)
- [Architecture](docs/architecture.md)
- [LLM choice ADR](docs/adr/001-llm-choice.md)
- [Third-party licenses](THIRD_PARTY_LICENSES.md)


## Self Note

- Inspired by [PageIndex](https://github.com/VectifyAI/PageIndex) and [OpenKB](https://github.com/VectifyAI/OpenKB), but **100% offline — no cloud, no API keys, no telemetry**.
