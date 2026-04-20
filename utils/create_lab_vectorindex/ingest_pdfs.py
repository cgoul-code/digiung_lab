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

    logging.info("All %d files verified on disk.", len(entries))
    return entries

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

def load_entry_as_documents(entry: dict) -> list[Document]:
    """
    Read all pages from the PDF described by a single document_store entry
    and attach all metadata fields to every page document.
    """
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

        # Base file metadata
        page_doc.metadata["source_file"] = str(pdf_path)
        page_doc.metadata["filename"]    = pdf_path.name

        # All fields from document_store.json
        page_doc.metadata["tittel"]            = entry.get("tittel") or ""
        page_doc.metadata["publisert_arstall"] = entry.get("publisert_arstall")
        page_doc.metadata["publisert_av"]      = (entry.get("publisert_av") or "").strip()
        page_doc.metadata["type_kilde"]        = entry.get("type_kilde") or ""
        page_doc.metadata["malgruppe"]         = entry.get("malgruppe") or ""
        page_doc.metadata["antall_deltakere"]  = entry.get("antall_deltakere") or ""
        page_doc.metadata["segment"]           = entry.get("segment") or ""
        page_doc.metadata["oppsummering"]      = entry.get("oppsummering") or ""

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
        pdf_path = str(Path(entry["filnavn"].replace("\\", os.sep)))

        if pdf_path in processed:
            logging.info("Skip (already ingested): %s", pdf_path)
            continue

        logging.info("Ingesting: %s", entry.get("tittel", pdf_path))
        try:
            pages = load_entry_as_documents(entry)
            logging.info("  → %d pages loaded", len(pages))
            _upsert_docs_into_index(name=name, storage=storage, documents=pages)
            processed.add(pdf_path)
            _save_processed_doc_ids(storage, name, processed)
            logging.info("  ✓ Done")
        except Exception:
            logging.error("  ✗ Failed for %s; will retry on next run", pdf_path, exc_info=True)

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
