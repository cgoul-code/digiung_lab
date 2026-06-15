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

## Redigerbare prompts per rapporttype

Hver analysetype (`query_type`: `problems`, `moments`, `personas`, `free`,
`strategisk_risiko`) har fire prompt-felter som kan redigeres fra klienten:
`extract_system`, `extract_prompt`, `aggregate_system`, `aggregate_prompt`.

### Standard + overstyring

Kode-standardene ligger i `QUERY_TYPES` i `aggregate_workflow.py` og er alltid
fasit. Redigeringer lagres som **overstyringer oppå** standardene — kun feltene
som faktisk avviker — i `query_type_prompts.json`:

```
./utils/create_lab_vectorindex/query_type_prompts.json   # lokal sti (PROMPTS_STORE_PATH)
query_type_prompts.json                                  # blob-rot (kun i Azure)
```

De strukturelle feltene (`output_key`, `structured`) kan **ikke** endres — de er
bundet til koden og rapport-/klientrendringen.

### Slik virker det

- **Redigere/lagre:** klient → `PUT /admin/query-types/<query_type>` med de fire
  feltene. Felter som er tomme eller lik standarden lagres ikke (holder
  overstyringsfila minimal).
- **Tilbakestille:** klient → `DELETE /admin/query-types/<query_type>` sletter
  overstyringen, slik at kode-standarden gjelder igjen.
- **Lesing:** `GET /query-types` returnerer alltid de *effektive* promptene
  (standard + overstyringer). Analysen bruker samme effektive konfigurasjon, som
  leses ferskt for hvert kall — ingen server-restart nødvendig.
- **Plassholdere må beholdes:** `{question}`, `{tittel}`, `{context}` i
  extract-promptene og `{question}`, `{n_docs}`, `{all_findings}` (+ `{n_personas}`
  for personas) i aggregate-promptene. Fjernes en påkrevd plassholder, feiler
  `.format()` ved kjøring. (Det er foreløpig ingen server-side validering av dette.)

I klienten finnes editoren under **Avansert → «Rediger prompts»** i aggregate-modus.

### Lokal vs. Azure-lagring

- **Lokalt** (standard, `WEBSITE_SITE_NAME` ikke satt): lagring skriver kun til
  den lokale `query_type_prompts.json`. Blob-speiling er en no-op. Fila dukker
  opp i `git status` — bestem om den skal committes (delte standard-overstyringer)
  eller ligge i `.gitignore` (kun lokal eksperimentering).
- **Azure** (eller lokalt med `USE_BLOB_INDEXES=true`): fila lastes ned fra
  blob-containeren ved oppstart, og hver lagring speiles tilbake til blob. Med
  `USE_BLOB_INDEXES=true` lokalt redigerer du altså **produksjonsdataene**.

Her er kommandoen. document_store.json lastes opp som én fil til blob-roten, og koden har allerede en hjelper for det:
.\.venv\Scripts\python.exe -c "from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv()); import os; os.environ['USE_BLOB_INDEXES']='true'; import azure_blob; print('container=', azure_blob.CONTAINER_NAME, 'enabled=', azure_blob.ENABLED); print('OK opplastet' if azure_blob.upload_document_store('./utils/create_lab_vectorindex/document_store.json') else 'FEILET — sjekk CONNECTION_STRING/CONTAINER_NAME')"
