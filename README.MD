## running for index build  
./.venv/Scripts/python.exe -m utils.create_lab_vectorindex.ingest_pdfs --name DigiUng_lab --document_store ./utils/create_lab_vectorindex/document_store.json
./.venv/Scripts/python.exe -m utils.create_lab_vectorindex.ingest_pdfs --name PSA_SSA --document_store ./utils/create_lab_vectorindex/document_store.json 

## remove documents not listed:
./.venv/Scripts/python.exe -m utils.create_lab_vectorindex.prune_index --name DigiUng_lab --dry-run #fires
./.venv/Scripts/python.exe -m utils.create_lab_vectorindex.prune_index --name DigiUng_lab

## Azure deployment

The server auto-detects Azure App Service via the `WEBSITE_SITE_NAME` env var.
When in Azure, indexes, source files, and `document_store.json` are mirrored
to the blob container set by `CONTAINER_NAME` (using `CONNECTION_STRING`).

Container layout:
- `<index_name>/` — LlamaIndex persist files (downloaded at startup, uploaded after reindex)
- `data/<index_name>/<filename>` — uploaded source PDFs/PPTX
- `document_store.json` — master JSON (uploaded after admin CRUD)

Bootstrap order at startup (Azure only):
1. Download `document_store.json` from container root.
2. For each index in the JSON: download `<idx>/` and `data/<idx>/` into the local
   working copy (`INDEX_STORAGE`, `DATA_DIR`).
3. Load indexes from local disk as usual.

Locally (no `WEBSITE_SITE_NAME`), the SDK path is skipped entirely by default —
files stay on disk and no blob traffic happens.

### Velg lokal eller server-indeks lokalt

Override-variabel `USE_BLOB_INDEXES` styrer hvilke indekser som brukes:

```
USE_BLOB_INDEXES=false      # default lokalt — bruk lokale filer
USE_BLOB_INDEXES=true       # kjør lokalt som om vi var i Azure:
                            #   - last ned alle indekser fra containeren ved oppstart
                            #   - admin-skriv (add/edit/delete/upload/reindex)
                            #     pushes til containeren
```

Container vinner alltid ved oppstart, så `true` overskriver eventuelle lokale
endringer du har gjort siden forrige nedlasting. Sett `false` (eller fjern
linja) for å gå tilbake til ren lokal modus.

Required env vars on App Service:
- `CONNECTION_STRING` — storage account connection string
- `CONTAINER_NAME` — blob container name (e.g. `chatbot`)
- plus the existing Azure OpenAI vars