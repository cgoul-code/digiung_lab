# utils

Hjelpescripts for å bygge / inspisere vektorindekser brukt av `query_server`.

Alle scripts kjøres fra repo-roten (`digiung_lab-query_server/`) slik at
relative stier (`./blobstorage/...`, `./utils/...`) treffer riktig:

```powershell
& .\.venv\Scripts\python.exe .\utils\<script>.py
```

---

## xlsx_to_doc_store.py

Konverterer en `kilder.xlsx` (Excel) til JSON i `document_store.json`-format
(et dictionary med indeksnavn som nøkkel og en liste med dokumentposter som
verdi). Brukes når en ny indeks skal seedes fra et regneark.

| Inn | Ut |
|---|---|
| `<XLSX_PATH>` (sett øverst i scriptet) | `./utils/create_lab_vectorindex/document_store_kilder.json` |

Endre `XLSX_PATH`, `OUT_PATH`, `INDEX_NAME` og `FILNAVN_PREFIX` øverst i fila
før kjøring. Kolonneoverskriftene i regnearket må matche nøklene i `COLMAP`
(blant annet `Tittel`, `Filnavn`, `Publisert årstall`, `URL`).

---

## hvaerinnafor-analyse

`_hvaerinnafor_clusters.py` leser TextNodes fra
`blobstorage/chatbot/hvaerinnafor/docstore.json` og produserer et snitt av
innholdet:

### `_hvaerinnafor_clusters.py`  →  diskriminerende termer per kategori

Grupperer chunks i 11 URL-baserte klynger og printer de mest distinktive
unigrammene + bigrammene per klynge til konsollet. Skriver ingen filer —
brukes til ad hoc inspeksjon når du vil se hvilke ord som er karakteristiske
for en gitt kategori.

> **Flyttet:** `_hvaerinnafor_extract.py` (rå-tekst + frekvensstatistikk) og
> `_hvaerinnafor_categories.py` (endelig kategori-JSON) er flyttet til
> `vectorindex_editor_server/utils/create_HEI_vectorindex/` og driftes nå
> derfra via HEI-regenereringspipelinen (`run_hei_regen.ps1`).

---

## create_lab_vectorindex/

Inneholder produksjonsscripts (`ingest_pdfs.py`, `document_store.json` osv.)
for å bygge selve LlamaIndex-vektorindeksen. Se inline-docstring i
`ingest_pdfs.py` for bruk.
