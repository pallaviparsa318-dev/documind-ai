from io import BytesIO
from pypdf import PdfWriter
import pytest
from documind.core import chunk_text, extract_pages, answer_question


def test_chunk_overlap_and_reconstruction():
    text = "abcdefghijklmnopqrstuvwxyz"
    assert chunk_text(text, chunk_size=10, overlap=3) == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=5, overlap=5)


def test_empty_document():
    assert chunk_text("  ") == []


def test_pdf_page_extraction():
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = BytesIO()
    writer.write(buffer)
    pages = extract_pages(buffer.getvalue(), "policy.pdf")
    assert pages == []  # Blank PDF has no extractable text.


def test_no_passages_no_hallucination():
    assert "No indexed documents" in answer_question("hello", [])


def test_local_vector_index(tmp_path):
    from documind.core import collection_at
    index = collection_at(str(tmp_path / "vectors"))
    index.upsert(ids=["a", "b"], documents=["cat", "dog"], metadatas=[{"source": "a", "page": 1}, {"source": "b", "page": 2}], embeddings=[[1.0, 0.0], [0.0, 1.0]])
    assert index.count() == 2
    result = index.query([[1.0, 0.0]], n_results=1)
    assert result["documents"][0] == ["cat"]
    assert collection_at(str(tmp_path / "vectors")).count() == 2
    index.delete(["a", "b"])
    assert index.count() == 0
