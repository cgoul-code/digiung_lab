"""Azure Blob Storage sync for digiung_lab.

When the app runs on Azure App Service (WEBSITE_SITE_NAME env var present),
the container holds the canonical copy of the data. The local filesystem is
treated as an ephemeral working copy.

Container layout (mirrors local paths):
    <idx>/                       ← LlamaIndex persist files (docstore.json, etc.)
    data/<idx>/<filename>        ← uploaded source files (PDF/PPTX)
    document_store.json          ← master JSON config (all indexes)

Local layout (the working copy):
    <INDEX_STORAGE>/<idx>/...
    <DATA_DIR>/<idx>/<filename>
    <DOCUMENT_STORE_PATH>

Sync points:
    - startup           → download_all()
    - file upload       → upload_data_file()
    - admin CRUD        → upload_document_store()
    - reindex finished  → upload_index_dir()
"""

import logging
import os
from pathlib import Path
from typing import Optional


IN_AZURE = bool(os.environ.get("WEBSITE_SITE_NAME") or os.environ.get("FUNCTIONS_WORKER_RUNTIME"))
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")

# Override toggle. Default: auto (sync only when running in Azure).
#   USE_BLOB_INDEXES=true   → always sync (e.g. when developing locally
#                              against the production container).
#   USE_BLOB_INDEXES=false  → never sync, even in Azure (rare; mostly
#                              for offline testing).
_override = os.getenv("USE_BLOB_INDEXES", "").strip().lower()
if _override in ("true", "1", "yes", "on"):
    _wants_sync = True
elif _override in ("false", "0", "no", "off"):
    _wants_sync = False
else:
    _wants_sync = IN_AZURE

ENABLED = bool(_wants_sync and CONNECTION_STRING and CONTAINER_NAME)

_log = logging.getLogger(__name__)
_log.info(
    "[azure_blob] IN_AZURE=%s  USE_BLOB_INDEXES=%r  ENABLED=%s  container=%s",
    IN_AZURE, _override or "auto", ENABLED, CONTAINER_NAME or "(unset)",
)


def _container_client():
    """Return a ContainerClient, or None if sync isn't enabled / configured."""
    if not ENABLED:
        return None
    from azure.storage.blob import BlobServiceClient
    bsc = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    return bsc.get_container_client(CONTAINER_NAME)


# ── primitives ────────────────────────────────────────────────────────────────

def download_prefix(prefix: str, local_dir: str) -> int:
    """Download every blob whose name starts with `prefix` into local_dir,
    preserving the relative path beneath the prefix.
    Returns the number of files written."""
    c = _container_client()
    if c is None:
        return 0
    os.makedirs(local_dir, exist_ok=True)
    n = 0
    for blob in c.list_blobs(name_starts_with=prefix):
        rel = blob.name[len(prefix):].lstrip("/")
        if not rel:
            continue
        local_path = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(c.download_blob(blob.name).readall())
        n += 1
    if n:
        _log.info("[azure_blob] Downloaded %d files from prefix %r → %s", n, prefix, local_dir)
    return n


def upload_dir(local_dir: str, prefix: str) -> int:
    """Upload every regular file in local_dir to <prefix>/<rel_path>.
    Returns the number of files uploaded."""
    c = _container_client()
    if c is None:
        return 0
    base = Path(local_dir)
    if not base.is_dir():
        return 0
    n = 0
    prefix = prefix.rstrip("/")
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(base).as_posix()
        blob_name = f"{prefix}/{rel}" if prefix else rel
        with open(path, "rb") as data:
            c.get_blob_client(blob_name).upload_blob(data, overwrite=True)
        n += 1
    if n:
        _log.info("[azure_blob] Uploaded %d files from %s → prefix %r", n, local_dir, prefix)
    return n


def download_file(blob_name: str, local_path: str) -> bool:
    c = _container_client()
    if c is None:
        return False
    try:
        data = c.download_blob(blob_name).readall()
    except Exception as e:
        _log.warning("[azure_blob] Could not download %s: %s", blob_name, e)
        return False
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(data)
    _log.info("[azure_blob] Downloaded %s → %s", blob_name, local_path)
    return True


def upload_file(local_path: str, blob_name: str) -> bool:
    c = _container_client()
    if c is None or not os.path.isfile(local_path):
        return False
    with open(local_path, "rb") as f:
        c.get_blob_client(blob_name).upload_blob(f, overwrite=True)
    _log.info("[azure_blob] Uploaded %s → %s", local_path, blob_name)
    return True


def delete_blob(blob_name: str) -> bool:
    c = _container_client()
    if c is None:
        return False
    try:
        c.delete_blob(blob_name)
        _log.info("[azure_blob] Deleted blob %s", blob_name)
        return True
    except Exception as e:
        _log.warning("[azure_blob] Could not delete %s: %s", blob_name, e)
        return False


def delete_prefix(prefix: str) -> int:
    """Delete every blob whose name starts with `prefix`. Returns count deleted."""
    c = _container_client()
    if c is None:
        return 0
    n = 0
    for blob in c.list_blobs(name_starts_with=prefix):
        try:
            c.delete_blob(blob.name)
            n += 1
        except Exception as e:
            _log.warning("[azure_blob] Could not delete %s: %s", blob.name, e)
    if n:
        _log.info("[azure_blob] Deleted %d blobs under prefix %r", n, prefix)
    return n


# ── high-level helpers used by query_server ───────────────────────────────────

def bootstrap_download(
    index_storage: str,
    data_dir: str,
    document_store_path: str,
) -> None:
    """At startup in Azure: pull master JSON, then every index folder + its
    source-file folder, into the ephemeral local working copy."""
    if not ENABLED:
        return
    _log.info("[azure_blob] Bootstrap download from container=%s (IN_AZURE=True)", CONTAINER_NAME)

    # 1. master document_store.json
    download_file("document_store.json", document_store_path)

    # 2. each index referenced in the master JSON
    if os.path.isfile(document_store_path):
        import json as _json
        try:
            with open(document_store_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
        except Exception as e:
            _log.error("[azure_blob] Could not read %s: %s", document_store_path, e)
            data = {}
        if isinstance(data, dict):
            for idx in data.keys():
                download_prefix(f"{idx}/", os.path.join(index_storage, idx))
                download_prefix(f"data/{idx}/", os.path.join(data_dir, idx))


def upload_index_dir(index_storage: str, index_name: str) -> int:
    """Upload all files under <index_storage>/<index_name>/ to blob <index_name>/."""
    local_dir = os.path.join(index_storage, index_name)
    return upload_dir(local_dir, index_name)


def upload_data_file(local_path: str, index_name: str, filename: str) -> bool:
    """Upload a single source file to blob data/<index_name>/<filename>."""
    return upload_file(local_path, f"data/{index_name}/{filename}")


def upload_document_store(document_store_path: str) -> bool:
    return upload_file(document_store_path, "document_store.json")
