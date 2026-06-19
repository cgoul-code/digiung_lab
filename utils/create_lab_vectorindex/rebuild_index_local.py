"""Rebuild a single index locally and push it to the production blob container.

Use this when a source URL is reachable from your machine but blocked
(HTTP 403) from the Azure datacenter IP — e.g. regjeringen.no behind a WAF.
The admin-menu "Regenerer" runs server-side and can never fetch such URLs;
this script runs the exact same ingest from your (non-blocked) IP and uploads
the finished index back to the same container the server reads from.

Prerequisites (set in the environment BEFORE running):
    USE_BLOB_INDEXES=true          # force blob sync even though we're local
    CONNECTION_STRING=<prod conn>  # production storage account connection string
    CONTAINER_NAME=<prod container>
plus the AZURE_OPENAI_EMBEDDINGS_* vars that ingest_pdfs.py needs.

Run from the repo root, e.g.:
    python -m utils.create_lab_vectorindex.rebuild_index_local --name Testindeks --mode full
"""

import argparse
import logging
import os
import shutil
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Local working copies (mirror query_server.py's *_LOCAL paths).
INDEX_STORAGE = os.getenv("INDEX_STORAGE", "./blobstorage/chatbot")
DATA_DIR = os.getenv("DATA_DIR", "./data")
DOCUMENT_STORE_PATH = os.getenv("DOCUMENT_STORE_PATH", "./utils/create_lab_vectorindex/document_store.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild one index locally and upload to the prod blob.")
    parser.add_argument("--name", required=True, help="Index name (e.g. Testindeks)")
    parser.add_argument("--mode", choices=("full", "incremental"), default="full",
                        help="full: delete existing index first (default); incremental: only add new entries")
    args = parser.parse_args()
    name = args.name

    # Imported here so env vars are read at module import time.
    import azure_blob
    from utils.create_lab_vectorindex.ingest_pdfs import run_incremental_ingest

    if not azure_blob.ENABLED:
        logging.error(
            "Blob sync is DISABLED. Set USE_BLOB_INDEXES=true, CONNECTION_STRING and "
            "CONTAINER_NAME before running, or you'll only build locally without uploading."
        )
        return 1

    logging.info("Rebuilding index '%s' (mode=%s) against container '%s'",
                 name, args.mode, azure_blob.CONTAINER_NAME)

    # 1. Pull the canonical document_store.json (has the admin-created entry) + any data files.
    azure_blob.download_file("document_store.json", DOCUMENT_STORE_PATH)
    pulled = azure_blob.download_prefix(f"data/{name}/", os.path.join(DATA_DIR, name))
    if pulled:
        logging.info("Pulled %d data file(s) from blob", pulled)

    # 2. full mode: clear the old index locally and in the blob so stale chunks don't linger.
    if args.mode == "full":
        persist_dir = os.path.join(INDEX_STORAGE, name)
        if os.path.isdir(persist_dir):
            shutil.rmtree(persist_dir)
            logging.info("Deleted local index at %s", persist_dir)
        removed = azure_blob.delete_prefix(f"{name}/")
        if removed:
            logging.info("Deleted %d blob(s) under %s/", removed, name)

    # 3. Ingest — runs from THIS machine's IP, so 403-blocked URLs download fine.
    run_incremental_ingest(
        storage=INDEX_STORAGE,
        name=name,
        document_store_path=DOCUMENT_STORE_PATH,
        data_dir=DATA_DIR,
        on_progress=lambda ev: logging.info("  %s", ev),
    )

    # 4. Verify something was actually built before uploading.
    persist_dir = os.path.join(INDEX_STORAGE, name)
    if not os.path.isfile(os.path.join(persist_dir, "docstore.json")):
        logging.error("No index produced for '%s' — every source failed or was empty. Nothing uploaded.", name)
        return 1

    # 5. Push the finished index back to the container the server serves from.
    uploaded = azure_blob.upload_index_dir(INDEX_STORAGE, name)
    logging.info("Uploaded %d index file(s) to blob '%s/'. Done.", uploaded, name)
    logging.info("Restart the app (or it will pick up the new index on next bootstrap).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
