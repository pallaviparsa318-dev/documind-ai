# DocuMind AI 📚

[![Tests](https://github.com/pallaviparsa318-dev/documind-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/pallaviparsa318-dev/documind-ai/actions/workflows/tests.yml)


**A local, source-grounded PDF question-answering application.** Built with Python, Streamlit, Ollama, Llama 3.2, `nomic-embed-text`, NumPy vector index and pypdf.

## Project highlights

DocuMind is a **fully local Retrieval-Augmented Generation (RAG) prototype** that lets users explore text-based PDF documents without sending their contents to a hosted language-model API.

| Area | Implementation |
| --- | --- |
| Interface | Streamlit PDF upload and question-answering UI |
| Parsing | `pypdf` with page-aware text extraction |
| Retrieval | Overlapping chunks, `nomic-embed-text` embeddings, cosine-similarity search in a persistent NumPy index |
| Generation | Ollama with Llama 3.2 and retrieved source excerpts |
| Quality checks | `pytest` unit tests run through GitHub Actions |

## Demo / screenshot

![DocuMind AI application demo showing local PDF question answering](Screenshot%202026-10-08%20143457.png)

*Actual local Streamlit app showing Ollama status, indexed document chunks, and a page-referenced answer. The screenshot also contains an earlier no-documents message from before indexing.*

## What it does

- Upload one or more text-based PDF documents.
- Extract text with original page numbers and split into overlapping chunks.
- Generate semantic embeddings locally using Ollama.
- Store vectors in persistent NumPy vector index and retrieve the top matching passages.
- Ask a local LLM to answer using only retrieved excerpts and cite `[Source N]`.
- Inspect the original filename, page number and passage for each retrieved source.
- Clear the local document index when needed.

**Important:** Citations are LLM-generated labels linked to retrieved excerpts; they are not independently verified. Always inspect the passages. Image-only PDFs are not supported without OCR.

## Architecture

```text
PDF upload -> pypdf -> page-aware chunks -> Ollama embeddings -> NumPy vector index
                                                                |
Question -> Ollama question embedding -> top-k vector retrieval --+
                                                                |
                                 retrieved passages -> Llama 3.2 -> answer + sources
```

## Prerequisites

- Python 3.10+ (Python 3.11 recommended)
- [Ollama](https://ollama.com/) installed and running
- Recommended: at least 8 GB RAM; close heavy apps while using local models

Download the local models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Run locally

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Python 3.13 on Windows can use this setup; no PyMuPDF DLL required.

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501). Upload a PDF, click **Index PDFs**, then ask a question.

## Test

```bash
python -m pytest -q
```

Unit tests cover chunking, input validation, page-number preservation and empty-context behavior. These tests do **not** validate LLM factual accuracy or retrieval quality.

## Evaluation plan (to complete before showcasing results)

Create a test set of at least 15 question/expected-source pairs from a non-sensitive PDF. Measure:

- **Recall@4:** fraction of questions whose gold supporting page appears among top 4 passages.
- **Answer groundedness:** manually assess whether each generated factual claim is supported by the retrieved excerpts.
- **Unanswerable questions:** test whether the system appropriately abstains when the PDF lacks the answer.

Publish the measured numbers only after running the evaluation. No performance results are claimed yet.

## Limitations

- No OCR for scanned PDFs.
- Local CPU inference may be slow on an 8 GB laptop.
- NumPy index persists on the machine in `.vectors/`, which is excluded from Git.
- PDF uploads are not access-controlled; **do not deploy this development UI publicly**.
- Re-indexing the same PDF is idempotent; differently named files with identical bytes share an index entry.
- No SQL tool or agent orchestration yet; this is a RAG system, not a multi-tool autonomous agent.
- Retrieval uses semantic similarity without a relevance threshold; irrelevant passages can still be returned.

## Future improvements

- Retrieval benchmark with published metrics, source-verification checks, OCR, hybrid retrieval, secure login and a read-only SQL tool.

## CV description (only after you run and understand it)

> Built a local retrieval-augmented PDF assistant using Python, Ollama, NumPy vector index and Streamlit; implemented page-aware chunking, semantic vector retrieval and source-linked question answering, with automated unit tests and CI.

## License

MIT. See `LICENSE`.
