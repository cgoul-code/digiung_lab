---
name: storage-architecture
description: Where digiung_lab production indexes actually live (Azure Files share, not Blob)
metadata:
  type: project
---

digiung_lab production indexes live in an **Azure Files SMB share** named `labdocumentquery`
(storage account `helsesvar`), mounted into the container at `INDEX_STORAGE` (`/blobstorage/chatbot`)
and read/written as ordinary files. Share contents: `DigiUng_lab/`, `Strategisk_risiko/`,
`CGs_kule_index/`, `cgs_kul_index/`, plus `document_store.json`.

The Blob-sync mechanism in `azure_blob.py` (`USE_BLOB_INDEXES` / `CONNECTION_STRING` /
`CONTAINER_NAME`) is a **separate, unused-for-digiung_lab path**. The `chatbot` Blob *container*
on the same `helsesvar` account belongs to the **llama_chatbot** app (indexes `hvaerinnafor*`,
`alleveiledere`) — NOT digiung_lab. Pointing the digiung_lab server at that blob container gives
`BlobNotFound` for `document_store.json` and risks cross-contaminating the two apps' data.

**Why:** The container name `chatbot` is reused across two apps in different storage backends
(Blob vs Files), which is easy to confuse. **How to apply:** To run the local server against real
prod data, make the file share's files available at `./blobstorage/chatbot` (mount+junction, or a
local copy) and keep `USE_BLOB_INDEXES=false`. Don't use the blob vars for digiung_lab.
Note: locally the code forces `INDEX_STORAGE = "." + ...`, so data must sit under `./blobstorage/chatbot`.
