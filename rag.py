"""Two vector collections in one ChromaDB:

  annual_reports  - knowledge: chunks of company annual reports, tagged company/year/page
  user_memory     - memory: short facts the agent saved about each user, tagged user_id
"""
import time
import uuid
from pathlib import Path

import chromadb
import pymupdf

DB_DIR = Path(__file__).parent / "chroma_db"

_client = chromadb.PersistentClient(path=str(DB_DIR))
knowledge = _client.get_or_create_collection("annual_reports")
memory = _client.get_or_create_collection("user_memory")


# ---------- knowledge: annual reports ----------

def pdf_pages(source):
    """Yield (page_number, text) from a PDF given as a path or raw bytes."""
    if isinstance(source, (bytes, bytearray)):
        doc = pymupdf.open(stream=source, filetype="pdf")
    else:
        doc = pymupdf.open(source)
    with doc:
        for i, page in enumerate(doc, start=1):
            yield i, page.get_text()


def chunk_text(text, size=1200, overlap=200):
    text = " ".join(text.split())
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def add_report(company, year, pages, source="", progress=None):
    """Chunk and store one report. `pages` is an iterable of (page_number, text).

    Re-ingesting the same company/year replaces the old chunks.
    Returns the number of chunks stored.
    """
    company, year = company.strip().upper(), int(year)
    docs, metas = [], []
    for page_no, text in pages:
        for chunk in chunk_text(text):
            docs.append(chunk)
            metas.append({"company": company, "year": year, "page": page_no, "source": source})
    if not docs:
        raise ValueError("No text found in this document (a scanned PDF needs OCR first).")

    knowledge.delete(where={"$and": [{"company": company}, {"year": year}]})
    batch = 100
    for start in range(0, len(docs), batch):
        end = start + batch
        knowledge.add(
            documents=docs[start:end],
            metadatas=metas[start:end],
            ids=[f"{company}_{year}_{i}" for i in range(start, min(end, len(docs)))],
        )
        if progress:
            progress(min(end, len(docs)), len(docs))
    return len(docs)


def list_reports():
    """Sorted list of (company, year) pairs that are indexed."""
    metas = knowledge.get(include=["metadatas"])["metadatas"]
    return sorted({(m["company"], m["year"]) for m in metas})


def delete_report(company, year):
    knowledge.delete(where={"$and": [{"company": company}, {"year": int(year)}]})


def search_reports(query, company, year=None, k=5):
    """Top-k chunks for `query`, restricted to one company (and optionally one year)."""
    company = company.strip().upper()
    where = {"company": company}
    if year:
        where = {"$and": [{"company": company}, {"year": int(year)}]}
    res = knowledge.query(query_texts=[query], n_results=k, where=where)
    return [
        {"text": doc, "company": meta["company"], "year": meta["year"], "page": meta["page"] or None}
        for doc, meta in zip(res["documents"][0], res["metadatas"][0])
    ]


# ---------- memory: facts about the user ----------

def save_memory(fact, user_id, replaces_id=None):
    """Store one fact. Pass replaces_id to overwrite an outdated memory."""
    if replaces_id:
        delete_memory(replaces_id, user_id)
    memory_id = uuid.uuid4().hex[:8]
    memory.add(
        documents=[fact],
        ids=[memory_id],
        metadatas=[{"user_id": user_id, "time": time.time()}],
    )
    return memory_id


def list_memories(user_id):
    res = memory.get(where={"user_id": user_id})
    rows = [
        {"id": i, "fact": d, "time": m["time"]}
        for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
    ]
    return sorted(rows, key=lambda r: r["time"])


def recall(query, user_id, k=5):
    """The k memories of this user most relevant to `query`."""
    total = len(memory.get(where={"user_id": user_id}, include=[])["ids"])
    if total == 0:
        return []
    res = memory.query(query_texts=[query], n_results=min(k, total), where={"user_id": user_id})
    return [
        {"id": i, "fact": d, "time": m["time"]}
        for i, d, m in zip(res["ids"][0], res["documents"][0], res["metadatas"][0])
    ]


def delete_memory(memory_id, user_id):
    # Check ownership so one user can never delete another user's memory.
    found = memory.get(ids=[memory_id], where={"user_id": user_id}, include=[])["ids"]
    if found:
        memory.delete(ids=found)
    return bool(found)
