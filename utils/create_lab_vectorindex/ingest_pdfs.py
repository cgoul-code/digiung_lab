"""
ingest_pdfs.py
==============
Drives ingestion from document_store.json — every entry in the JSON
must have a file that exists on disk, otherwise the script aborts.

Usage:
    python ingest_pdfs.py
    python ingest_pdfs.py --document_store ./utils/create_lab_vectorindex/document_store.json
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.readers.file import PDFReader, PptxReader

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlsplit, urlunsplit

# Cache for parent SPA pages keyed by URL without fragment
_SPA_PAGE_CACHE: dict[str, str] = {}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ── Embedding model ───────────────────────────────────────────────────────────

Settings.embed_model = AzureOpenAIEmbedding(
    model=os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL"),
    deployment_name=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
)

Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)

# ── Manifest helpers ──────────────────────────────────────────────────────────

MANIFEST_FILENAME = "_ingested_docs.json"

def _load_processed_doc_ids(storage: str, name: str) -> set[str]:
    path = os.path.join(storage, name, MANIFEST_FILENAME)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def _save_processed_doc_ids(storage: str, name: str, ids: set[str]) -> None:
    path = os.path.join(storage, name, MANIFEST_FILENAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f, ensure_ascii=False, indent=2)

# ── Document store ────────────────────────────────────────────────────────────

def _load_and_validate_document_store(json_path: str, index_name: str) -> list[dict]:
    """
    Load document_store.json and verify every file listed actually exists.
    Supports both the legacy flat-list format and the new dict-of-lists format:
      {"IndexName": [...], "OtherIndex": [...]}
    Aborts with a clear error message if any file is missing.
    """
    if not os.path.isfile(json_path):
        logging.error("document_store.json not found at: %s", json_path)
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        entries = data
    elif index_name in data:
        entries = data[index_name]
    else:
        logging.error(
            "Index '%s' not found in %s. Available: %s",
            index_name, json_path, list(data.keys())
        )
        sys.exit(1)

    logging.info("Loaded %d entries for index '%s' from %s", len(entries), index_name, json_path)

    missing = []
    for entry in entries:
        if entry.get("url"):
            continue
        filnavn = entry.get("filnavn", "")
        # Normalise Windows backslashes to the OS separator
        pdf_path = Path(filnavn.replace("\\", os.sep))
        if not pdf_path.is_file():
            missing.append(str(pdf_path))

    if missing:
        logging.error("The following files listed in document_store.json do not exist on disk:")
        for m in missing:
            logging.error("  ✗ %s", m)
        logging.error("Fix the missing files and rerun. Aborting.")
        sys.exit(1)

    logging.info("All %d entries verified.", len(entries))
    return entries


def _entry_key(entry: dict) -> str:
    """Stable identifier used for manifest tracking."""
    if entry.get("url"):
        return entry["url"]
    return str(Path(entry.get("filnavn", "").replace("\\", os.sep)))

# ── Index helpers ─────────────────────────────────────────────────────────────

def _upsert_docs_into_index(
    name: str,
    storage: str,
    documents: list[Document],
) -> VectorStoreIndex:
    persist_dir = os.path.join(storage, name)
    os.makedirs(persist_dir, exist_ok=True)

    if os.path.isfile(os.path.join(persist_dir, "docstore.json")):
        logging.info("Loading existing index from %s", persist_dir)
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        for doc in documents:
            index.insert(doc)
    else:
        logging.info("Creating new index at %s", persist_dir)
        index = VectorStoreIndex.from_documents(documents)

    index.storage_context.persist(persist_dir=persist_dir)
    return index

# ── PDF loading ───────────────────────────────────────────────────────────────

def _apply_entry_metadata(doc: Document, entry: dict, source: str, filename: str) -> None:
    doc.metadata["source_file"]       = source
    doc.metadata["filename"]          = filename
    doc.metadata["tittel"]            = entry.get("tittel") or ""
    doc.metadata["publisert_arstall"] = entry.get("publisert_arstall")
    doc.metadata["publisert_av"]      = (entry.get("publisert_av") or "").strip()
    doc.metadata["type_kilde"]        = entry.get("type_kilde") or ""
    doc.metadata["malgruppe"]         = entry.get("malgruppe") or ""
    doc.metadata["antall_deltakere"]  = entry.get("antall_deltakere") or ""
    doc.metadata["segment"]           = entry.get("segment") or ""
    doc.metadata["oppsummering"]      = entry.get("oppsummering") or ""


def _http_get(url: str) -> str:
    resp = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0 (digiung-lab-ingest)"},
    )
    resp.raise_for_status()
    if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def _spa_topic_text(url: str) -> tuple[str, str]:
    """
    For helsenorge.no/.../hvaerinnafor/#/temabeskrivelse/<slug> URLs the topic
    body is embedded as JSON inside the parent page. Fetch the parent (cached)
    and extract the matching topic object's title + ingress1 + ingress2.
    Returns (title, body_text).
    """
    parts = urlsplit(url)
    fragment = parts.fragment  # e.g. "/temabeskrivelse/<slug>"
    if not fragment.startswith("/"):
        fragment = "/" + fragment
    parent_url = urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

    html = _SPA_PAGE_CACHE.get(parent_url)
    if html is None:
        html = _http_get(parent_url)
        _SPA_PAGE_CACHE[parent_url] = html

    needle = f'"urlPath":"{fragment}"'
    idx = html.find(needle)
    if idx < 0:
        raise ValueError(f"Topic {fragment} not found in parent page {parent_url}")

    # Walk backwards to the opening '{' of this object, then bracket-balance forward.
    start = html.rfind("{", 0, idx)
    if start < 0:
        raise ValueError(f"Could not locate topic object start for {fragment}")

    depth = 0
    in_str = False
    escape = False
    end = -1
    for i in range(start, len(html)):
        ch = html[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
    if end < 0:
        raise ValueError(f"Could not locate topic object end for {fragment}")

    obj = json.loads(html[start:end])
    title = (obj.get("title") or "").strip()
    ingress1 = (obj.get("ingress1") or "").strip()
    ingress2 = (obj.get("ingress2") or "").strip()
    body = "\n\n".join(p for p in (title, ingress1, ingress2) if p)
    if not body:
        raise ValueError(f"Topic {fragment} has no extractable text")
    return title, body


def _fetch_url_as_document(entry: dict) -> list[Document]:
    """Fetch an HTML page and return a single cleaned-text Document."""
    url = entry["url"]

    # SPA hash-fragment URLs (helsenorge "hvaerinnafor") need special handling
    fragment = urlsplit(url).fragment
    if fragment and "/temabeskrivelse/" in fragment:
        page_title, text = _spa_topic_text(url)
        enriched = dict(entry)
        if not enriched.get("tittel"):
            enriched["tittel"] = page_title
        doc = Document(text=text, doc_id=url)
        _apply_entry_metadata(doc, enriched, source=url, filename=url)
        doc.metadata["url"] = url
        return [doc]

    html = _http_get(url)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "aside"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    if not text.strip():
        raise ValueError(f"No textual content extracted from {url}")

    page_title = ""
    if soup.title and soup.title.string:
        page_title = soup.title.string.strip()
    elif soup.find("h1"):
        page_title = soup.find("h1").get_text(strip=True)

    enriched = dict(entry)
    if not enriched.get("tittel"):
        enriched["tittel"] = page_title

    doc = Document(text=text, doc_id=url)
    _apply_entry_metadata(doc, enriched, source=url, filename=url)
    doc.metadata["url"] = url
    return [doc]


def load_entry_as_documents(entry: dict) -> list[Document]:
    """
    Build Document(s) for a single document_store entry. Supports two formats:
      - File entry: {"filnavn": "...pdf", ...} → one Document per page
      - URL entry:  {"url": "https://...", "method": "GET"} → one Document per URL
    """
    if entry.get("url"):
        return _fetch_url_as_document(entry)

    filnavn = entry.get("filnavn", "")
    pdf_path = Path(filnavn.replace("\\", os.sep))

    suffix = pdf_path.suffix.lower()
    if suffix in (".pptx", ".ppt"):
        reader = PptxReader()
    else:
        reader = PDFReader()
    pages: list[Document] = reader.load_data(file=pdf_path)

    for page_doc in pages:
        page_doc.doc_id = str(pdf_path)
        _apply_entry_metadata(page_doc, entry, source=str(pdf_path), filename=pdf_path.name)

    return pages

# ── Main incremental ingestion ────────────────────────────────────────────────

def run_incremental_ingest(
    storage: str,
    name: str,
    document_store_path: str,
):
    # Load + validate — fails here if any file is missing
    entries = _load_and_validate_document_store(document_store_path, name)

    processed = _load_processed_doc_ids(storage, name)
    logging.info("Found %d entries in document store, %d already ingested", len(entries), len(processed))

    for entry in entries:
        key = _entry_key(entry)

        if key in processed:
            logging.info("Skip (already ingested): %s", key)
            continue

        logging.info("Ingesting: %s", entry.get("tittel") or key)
        try:
            pages = load_entry_as_documents(entry)
            logging.info("  → %d document(s) loaded", len(pages))
            _upsert_docs_into_index(name=name, storage=storage, documents=pages)
            processed.add(key)
            _save_processed_doc_ids(storage, name, processed)
            logging.info("  ✓ Done")
        except requests.HTTPError as e:
            logging.warning("  ✗ Skipping %s: %s", key, e)
        except Exception:
            logging.error("  ✗ Failed for %s; will retry on next run", key, exc_info=True)

    logging.info("Ingestion complete. Total ingested: %d / %d", len(processed), len(entries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into a vector index")
    parser.add_argument("--storage",        default="./blobstorage/chatbot",                          help="Root storage folder for the index")
    parser.add_argument("--name",           default="DigiUng_lab",                                            help="Index name (subfolder under storage)")
    parser.add_argument("--document_store", default="./utils/create_lab_vectorindex/document_store.json", help="Path to document_store.json")
    args = parser.parse_args()

    run_incremental_ingest(
        storage=args.storage,
        name=args.name,
        document_store_path=args.document_store,
    )
