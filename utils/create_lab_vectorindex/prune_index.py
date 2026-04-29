"""
prune_index.py
==============
Removes documents from a vector index that are no longer listed in
document_store.json. Useful after deleting entries from the JSON to keep
the persisted index in sync.

Usage:
    python -m utils.create_lab_vectorindex.prune_index --name DigiUng_lab
    python -m utils.create_lab_vectorindex.prune_index --name PSA_SSA --dry-run
"""

import os
import json
import logging
import argparse
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Embedding model is required when loading the index even though we only delete
Settings.embed_model = AzureOpenAIEmbedding(
    model=os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL"),
    deployment_name=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
)

MANIFEST_FILENAME = "_ingested_docs.json"


def _expected_paths(document_store_path: str, index_name: str) -> set[str]:
    """Return the set of file paths that SHOULD be in the index."""
    with open(document_store_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        entries = data
    elif index_name in data:
        entries = data[index_name]
    else:
        raise SystemExit(f"Index '{index_name}' not found in {document_store_path}")
    paths = set()
    for entry in entries:
        filnavn = entry.get("filnavn", "")
        paths.add(str(Path(filnavn.replace("\\", os.sep))))
    return paths


def _load_manifest(persist_dir: str) -> set[str]:
    path = os.path.join(persist_dir, MANIFEST_FILENAME)
    if not os.path.isfile(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def _save_manifest(persist_dir: str, ids: set[str]) -> None:
    path = os.path.join(persist_dir, MANIFEST_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f, ensure_ascii=False, indent=2)


def prune(storage: str, name: str, document_store_path: str, dry_run: bool):
    persist_dir = os.path.join(storage, name)
    if not os.path.isfile(os.path.join(persist_dir, "docstore.json")):
        raise SystemExit(f"No index found at {persist_dir}")

    expected = _expected_paths(document_store_path, name)
    ingested = _load_manifest(persist_dir)
    stale    = ingested - expected

    logging.info("Index '%s': %d expected, %d ingested, %d stale", name, len(expected), len(ingested), len(stale))

    if not stale:
        logging.info("Nothing to prune.")
        return

    for p in sorted(stale):
        logging.info("  ✗ stale: %s", p)

    if dry_run:
        logging.info("--dry-run set; not modifying the index.")
        return

    logging.info("Loading index from %s ...", persist_dir)
    ctx = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(ctx)

    deleted = 0
    for doc_id in stale:
        try:
            # doc_id was set to str(pdf_path) at ingest time
            index.delete_ref_doc(doc_id, delete_from_docstore=True)
            deleted += 1
            logging.info("  ✓ removed from index: %s", doc_id)
        except Exception as e:
            logging.error("  ✗ failed to remove %s: %s", doc_id, e)

    index.storage_context.persist(persist_dir=persist_dir)

    _save_manifest(persist_dir, ingested - stale)
    logging.info("Pruned %d documents from '%s'.", deleted, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove stale documents from a vector index")
    parser.add_argument("--storage",        default="./blobstorage/chatbot")
    parser.add_argument("--name",           required=True, help="Index name (subfolder under storage)")
    parser.add_argument("--document_store", default="./utils/create_lab_vectorindex/document_store.json")
    parser.add_argument("--dry-run",        action="store_true", help="Show what would be removed without modifying the index")
    args = parser.parse_args()

    prune(
        storage=args.storage,
        name=args.name,
        document_store_path=args.document_store,
        dry_run=args.dry_run,
    )
