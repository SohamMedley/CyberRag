# CYBER-RAG

A small, fast retrieval-augmented generation (RAG) web app for your own
documents. Upload PDFs or text files, index them into a local FAISS vector
store, and ask questions that are answered **only** from what you uploaded.

Built with Flask, sentence-transformers, FAISS and the Groq API.

```
┌────────────┐   upload    ┌────────────┐   chunk + embed   ┌──────────────┐
│  Library   │ ──────────► │  uploads/  │ ────────────────► │  FAISS index │
│  (browser) │             └────────────┘                   └──────┬───────┘
└─────┬──────┘                                                    │ top-k
      │  ask                                          ┌───────────▼────────┐
      └────────────────────────────────────────────► │ Groq chat model    │
                                                     │ (grounded answer)  │
                                                     └────────────────────┘
```

## Features

- **Two views, nothing else:** *Library* (upload, index, per-file delete) and *Ask*.
- **Grounded answers** - the model is instructed to answer only from retrieved
  passages and to say so when the documents do not cover the question.
- **Citations** - every answer lists the documents (and page numbers) used.
- **Per-document delete** - remove a document from the vector store, disk and UI
  in one click, including re-indexing without duplicates (`replace` semantics).
- **Efficient Groq usage** - automatic model resolution, fallback models on rate
  limits, prompt caching friendly prompt layout, token budgets for context and
  history, and no API call at boot.
- **Deploy-ready** - `gunicorn.conf.py`, `Procfile` and `render.yaml` included;
  see [DEPLOY_RENDER.md](DEPLOY_RENDER.md).

## Requirements

- Python 3.11+
- A free Groq API key: <https://console.groq.com/keys>
- ~1 GB free disk for the PyTorch CPU wheels and the embedding model

## Quickstart (local)

```bash
git clone <your-repo-url> && cd CyberRag

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # ~5-10 min: installs CPU PyTorch

cp .env.example .env               # then edit .env and paste your key
#   GROQ_API_KEY=gsk_your_real_key

python app.py                      # -> http://127.0.0.1:5000
```

Prefer to keep secrets out of the repo? Put the real key in `.env.local`
(git-ignored, overrides `.env`) or export `GROQ_API_KEY` in your shell.

### Using the app

1. **Library tab** - drop `.pdf` / `.txt` files, review the queue, click
   *Index documents*. Each file is chunked (~350 words, 50 overlap), embedded
   with `all-MiniLM-L6-v2` and stored in `data/index/`.
2. **Ask tab** - type a question. Answers stream back with the source documents
   and the retrieved passages listed underneath. Follow-up questions include the
   last three turns of history, so you can say "and what about the labs?".
3. **Delete** - the trash icon next to a document removes it from the index, the
   uploads folder and the archive folder.

The sample documents in `Inputs & Ques Sample/` are handy for a first try.

## Configuration

Everything is optional except `GROQ_API_KEY`. Copy `.env.example` for the full
annotated list; the most useful knobs:

| Variable | Default | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | – | Groq key (`gsk_...`). Required before asking questions. |
| `GROQ_MODEL` | auto | Pin a model. Empty = pick the best available at runtime. |
| `GROQ_FALLBACK_MODELS` | `openai/gpt-oss-20b,llama-3.1-8b-instant` | Tried on rate limits/errors. |
| `GROQ_REASONING_EFFORT` | `low` | `low/medium/high` for gpt-oss reasoning models. |
| `TOP_K_RETRIEVAL` | `4` | Passages retrieved per question. |
| `MIN_SIMILARITY_SCORE` | `0.15` | Below this, a passage is considered irrelevant. |
| `MAX_CONTEXT_TOKENS` / `MAX_ANSWER_TOKENS` | `3000` / `700` | Token budgets. |
| `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` | `350` / `50` | Chunking. |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Swap for a better model if you have GPU/RAM. |
| `EMBEDDING_THREADS` | all cores | Set to `1` on shared/limited CPUs. |
| `WARMUP_EMBEDDINGS` | `false` | Load the model during boot instead of first use. |
| `DATA_DIR` / `UPLOAD_DIR` | `./data`, `./uploads` | Point at a persistent disk when hosting. |
| `PORT` | `5000` | Render and friends inject this. |

## HTTP API

| Method & path | Purpose |
| --- | --- |
| `GET /` | The web UI. |
| `GET /healthz` | Liveness probe (`{"status":"ok","chunks":N,"version":"..."}`). |
| `GET /status` | Library summary: documents, chunk counts, embedding/LLM model, limits. |
| `POST /upload` | `multipart/form-data` with one or more `files`. |
| `POST /index` | `{"filenames": [...]}` (omit to index everything staged). Add `"replace": true` to re-index files already in the library. |
| `POST /ask` | `{"question": "...", "history": [{"role","content"}, ...]}`. |
| `DELETE /documents/<filename>` | Remove a document (also `POST /documents/delete` with `{"filename": ...}`). |
| `POST /reset` | Clear the whole index and upload staging. |

Example:

```bash
curl -F "files=@report.pdf" http://127.0.0.1:5000/upload
curl -X POST -H 'Content-Type: application/json' \
     -d '{"question":"What is the attendance policy?"}' \
     http://127.0.0.1:5000/ask
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest                      # 100+ tests, ~1 s, no network, no API key
pytest -m slow              # opt-in: uses the real embedding model (downloads ~90 MB)
```

The fast suite replaces the embedder and the Groq client with deterministic
stubs, so it runs offline and never spends tokens.

## Project layout

```
app.py                  Flask application factory + routes
rag/config.py           Settings (env parsing, paths, limits)
rag/documents.py        PDF/text extraction and chunking
rag/embedder.py         sentence-transformers wrapper (thread-safe, lazy)
rag/store.py            FAISS index + chunk metadata, add/search/delete/reset
rag/llm.py              Groq client: model resolution, retries, prompt building
rag/services.py         Wires settings/store/embedder/llm together
templates/, static/     Library + Ask UI
tests/                  Pytest suite (stubbed, offline)
scripts/offline_demo.py Run the UI with no model download and no API calls
gunicorn.conf.py        Production server config (1 worker, threads)
render.yaml             Render blueprint
DEPLOY_RENDER.md        Deployment walkthrough
```

## Troubleshooting

- **`503 The embedding model is not available yet`** - the first index/ask needs
  to download the sentence-transformer model. Give the host internet access once
  or pre-populate `HF_HOME`.
- **`429` / rate-limit answers** - the free Groq tier has tokens-per-minute
  limits. Wait a moment or pin a lighter `GROQ_MODEL`.
- **`CYBER-RAG expects groq>=1.0`** - an outdated `groq` package is installed;
  reinstall with `pip install -r requirements.txt`.
- **Answers say "I could not find sufficient information..."** - the question
  did not match any passage above `MIN_SIMILARITY_SCORE`; try adding the
  document, rephrasing, or lowering the threshold slightly.

## License

Provided as-is for educational and personal use.
