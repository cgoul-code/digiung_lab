## running for index build  
./.venv/Scripts/python.exe -m utils.create_lab_vectorindex.ingest_pdfs --name DigiUng_lab --document_store ./utils/create_lab_vectorindex/document_store.json
./.venv/Scripts/python.exe -m utils.create_lab_vectorindex.ingest_pdfs --name PSA_SSA --document_store ./utils/create_lab_vectorindex/document_store.json 

## remove documents not listed:
./.venv/Scripts/python.exe -m utils.create_lab_vectorindex.prune_index --name DigiUng_lab --dry-run #fires
./.venv/Scripts/python.exe -m utils.create_lab_vectorindex.prune_index --name DigiUng_lab

## Azure deployment

> Moving off Azure? See [docs/fly-migration.md](docs/fly-migration.md) for a
> step-by-step migration to Fly.io (Azure OpenAI stays; only hosting moves).

The server auto-detects Azure App Service via the `WEBSITE_SITE_NAME` env var.
When in Azure, indexes, source files, and `document_store.json` are mirrored
to the blob container set by `CONTAINER_NAME` (using `CONNECTION_STRING`).

> Dette beskriver blob-mekanismen i koden. Dagens produksjonsoppsett lagrer i
> stedet varig tilstand på en montert SMB file share — se
> [App Service-oppsett](#app-service-oppsett-lagring-og-miljøvariabler) for
> hvilke variabler som faktisk må settes, og hvordan de to henger sammen.

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

### App Service-oppsett: lagring og miljøvariabler

**Filsystemet i Azure er efemert.** Oryx pakker ut deploy-pakka til `/tmp/<hash>`
og kjører prosessen derfra — CWD er altså *ikke* `/home/site/wwwroot`. Alt
utenfor `/home` og utenfor monterte volum slettes ved **hver containerstart**,
ikke bare ved deploy. Samtlige standardstier i `query_server.py` er relative
(`./utils/create_lab_vectorindex/…`) og lander dermed i `/tmp`.

Derfor må alle stivariablene under settes eksplisitt i App Settings. Gjør man
det ikke, ser lagring ut til å virke — endepunktet svarer 200, fila skrives,
GUI-et viser riktig — og så er alt tilbake til deploy-pakkas innhold ved neste
restart. Symptomet er analysemaler og bank→mal-koblinger som «ikke lagres», og
som flakker mellom ulike tilstander fra økt til økt.

#### Path mapping

`/blobstorage/chatbot` → SMB file share `labdocumentquery`
(**Configuration → Path mappings**).

Mountet finnes kun i app-containeren. Kudus web-console og filbehandler kjører i
en *egen* container som bare deler `/home`, og ser det ikke. Skal du inspisere
eller kopiere noe der, bruk **Development Tools → SSH**.

#### Stier — må settes, ellers går data tapt

| Variabel | Verdi | Uten den |
|---|---|---|
| `DOCUMENT_STORE_PATH` | `/blobstorage/chatbot/document_store.json` | `/tmp` — dokumentbankene forsvinner |
| `CUSTOM_QT_STORE_PATH` | `/blobstorage/chatbot/custom_query_types.json` | `/tmp` — egne analysemaler forsvinner |
| `INDEX_QT_STORE_PATH` | `/blobstorage/chatbot/index_query_types.json` | `/tmp` — koblingen bank→maler forsvinner |
| `PROMPTS_STORE_PATH` | `/blobstorage/chatbot/query_type_prompts.json` | `/tmp` — prompt-overstyringer forsvinner |
| `EXAMPLES_STORE_PATH` | `/blobstorage/chatbot/example_questions.json` | `/tmp` — forslagsspørsmålene forsvinner |
| `INDEX_STORAGE` | `/blobstorage/chatbot/indexes` | faller til `/blobstorage/chatbot` |
| `DATA_DIR` | `/blobstorage/chatbot/data` | faller til `/data` — efemer |

Alle sju er bekreftet satt i dagens App Settings.

#### Blob-speiling — valgfri, og den overstyrer file share-en

Koden har i tillegg en eldre speilingsmekanisme mot **Blob Storage** (ikke
samme tjeneste som file share-en; separate navnerom, også på samme konto):

| Variabel | Verdi |
|---|---|
| `CONNECTION_STRING` | connection string til lagringskontoen |
| `CONTAINER_NAME` | blob-container, f.eks. `chatbot` |
| `USE_BLOB_INDEXES` | `true` / `false` — uten den er den **på** i Azure |

`azure_blob.ENABLED` krever at *begge* de to første er satt. Mangler én, blir
hver `upload_file` en stille no-op: returverdien ignoreres av `_save_*`, så
ingenting feiler synlig.

⚠️ **Rekkefølgen ved oppstart er viktig.** `_load_all_indexes_async` laster ned
`SETTINGS_BLOBS` **over** de konfigurerte stiene. Peker de på file share-en, og
blob-speilingen er aktiv, vinner blob-innholdet over share-en hver gang
containeren starter.

Derfor står `USE_BLOB_INDEXES=0` i dagens oppsett: det gir `ENABLED=False`, og
hele blob-veien — både bootstrap-nedlastingen og speilingen av lagringer — er
av. Indeksene leses rett fra mountet i stedet. **Ikke fjern den variabelen.**
Uten den er standarden «på i Azure», og da overskrives file share-filene av
gamle blobs ved neste containerstart.

#### Azure OpenAI

| Variabel | Merknad |
|---|---|
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION` | felles |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | standardmodell |
| `AZURE_OPENAI_EXTRACT_DEPLOYMENT` | sterk modell, per-dokument-uttrekk |
| `AZURE_OPENAI_AGGREGATE_DEPLOYMENT` | rask/billig modell, oppsummering |
| `AZURE_OPENAI_EMBEDDINGS_ENDPOINT`, `_API_KEY`, `_API_VERSION`, `_MODEL`, `_DEPLOYMENT` | embeddings |

#### Øvrige

| Variabel | Standard |
|---|---|
| `CORS_ALLOWED_ORIGINS` | de to deployede klientene; komma-separert, uten skråstrek til slutt |
| `SIMILARITY_CUTOFF` | `0.3` |
| `MAX_UPLOAD_BYTES` | 100 MB |
| `FETCH_URL_MAX_BYTES` | 100 MB |

`WEBSITE_SITE_NAME` settes av plattformen og er det appen bruker til å avgjøre
at den kjører i Azure. Ikke sett den manuelt.

#### Verifisere oppsettet

I **Development Tools → SSH**:

```sh
env | grep -E 'STORE_PATH|INDEX_STORAGE|DATA_DIR|CONTAINER_NAME|USE_BLOB'
ls -l --time-style=full-iso /blobstorage/chatbot/*.json
curl -s localhost:8000/admin/index-query-types
```

`curl`-en skal vise bank→mal-koblingene. Kommer `{}`, leser appen et annet sted
enn du tror. I **Log stream** ved oppstart:

```
[config] LOCAL=False
[config] INDEX_STORAGE=…
[config] DOCUMENT_STORE_PATH (fallback)=…
[config] BLOB_SYNC: enabled=…  in_azure=…  container=…
```

Den virkelige testen er å lagre en analysemal på en dokumentbank i klienten og
se at tidsstempelet på `/blobstorage/chatbot/index_query_types.json` endrer seg.
At du selv kan laste opp filer til share-en fra portalen beviser bare dine egne
rettigheter — ikke at appen får skrive over SMB som sin egen bruker.

#### Merk: JSON-filene i `utils/create_lab_vectorindex/` er sporet i git

`custom_query_types.json`, `index_query_types.json`, `query_type_prompts.json`
og `example_questions.json` ligger i repoet og følger med hver deploy. Så lenge
stivariablene over er satt, er de bare seed-/utviklingsdata og rører ikke
produksjon. Er de *ikke* satt, er det disse filene som er produksjonstilstanden,
og enhver commit ruller produksjon tilbake.

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

For de innebygde malene kan de strukturelle feltene (`output_key`, `structured`)
**ikke** endres — de er bundet til koden. Egne maler bygget i veiviseren setter
dem selv; se neste avsnitt.

## Analysemaler fra veiviseren

En strukturert mal spør hvert dokument om et JSON-objekt og oppsummeringen om et
annet, i stedet for punktlister. Hvilke nøkler de objektene har er **data**, ikke
kode: malen deklarerer dem, og både workflow, klient og .docx-rapport leser
deklarasjonen. `strategisk_risiko` deklarerer analysekjeden sin på samme måte,
så innebygde og egendefinerte maler kjører samme kodevei.

Feltene i en malkonfigurasjon:

| felt | betyr |
| --- | --- |
| `structured` | svarene er JSON-objekter, ikke punktlister |
| `doc_fields` | listefeltene i dokumentsvaret, `[{key, label, beskrivelse}]` |
| `doc_notes` | enkeltverdi-feltene i dokumentsvaret (relevansvurdering o.l.) |
| `agg_items_key` | nøkkelen syntesen legger funnene under (veiviseren: `temaer`) |
| `agg_items_label` | hva ett funn heter i opptellingen («12 risikoområder») |
| `agg_item_fields` | listefeltene inne i hvert funn |
| `agg_top_fields` | listefeltene på tvers av funnene |
| `wizard` | svarene malen ble skrevet av, så den kan åpnes i veiviseren igjen |

Flagg på et felt: `lead` (vises før listen, ikke etter — i veiviseren bestemt av
om feltet står over eller under raden for listen), `context: false` (blir stående i
analysen per dokument, men sendes ikke til oppsummeringen), `tone` (strekfarge i
margen: `"#rrggbb"`, eller `"danger"` for rødt). Det øverste feltet i `doc_fields`
står for dokumentet der bare én liste får plass.

### Slik virker veiviseren

Promptene **skrives av svarene** på serveren, ikke av brukeren:

- `POST /admin/query-types/compose` med `{wizard: {...}}` returnerer de fire
  promptene uten å lagre noe — det er forhåndsvisningen i klienten.
- `POST /admin/query-types` og `PUT /admin/query-types/<qt>` med `wizard` i
  kroppen lagrer malen: samme komposisjon, pluss deklarasjonen over. Teksten man
  godkjenner i forhåndsvisningen er derfor teksten analysen kjører på.
- Brukerskrevet tekst får klammeparentesene doblet automatisk i
  `aggregate_system`, siden den går gjennom `.format()` før den sendes. Et
  JSON-eksempel i en instruksjon er noe modellen skal se, ikke en plassholder.

Egne maler lagres hele i `custom_query_types.json` (`CUSTOM_QT_STORE_PATH`), og
speiles til blob som de andre lagrene.

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
Nye maler lages under **Administrer analysemaler**: «Ny analysemal (veiviser)»
bygger en strukturert mal, «Enkel mal» to instruksjoner skrevet for hånd. En mal
laget i veiviseren åpnes i veiviseren igjen når man trykker «Rediger».

### Lokal vs. Azure-lagring

- **Lokalt** (standard, `WEBSITE_SITE_NAME` ikke satt): lagring skriver kun til
  den lokale `query_type_prompts.json`. Blob-speiling er en no-op. Fila dukker
  opp i `git status` — bestem om den skal committes (delte standard-overstyringer)
  eller ligge i `.gitignore` (kun lokal eksperimentering).
- **Azure** (eller lokalt med `USE_BLOB_INDEXES=true`): fila lastes ned fra
  blob-containeren ved oppstart, og hver lagring speiles tilbake til blob. Med
  `USE_BLOB_INDEXES=true` lokalt redigerer du altså **produksjonsdataene**.

Hvor fila ligger i Azure styres av `PROMPTS_STORE_PATH`. Er den ikke satt,
havner den i `/tmp` og forsvinner ved hver containerstart — gjelder tilsvarende
for `CUSTOM_QT_STORE_PATH`, `INDEX_QT_STORE_PATH` og `EXAMPLES_STORE_PATH`. Se
[App Service-oppsett](#app-service-oppsett-lagring-og-miljøvariabler).

Her er kommandoen. document_store.json lastes opp som én fil til blob-roten, og koden har allerede en hjelper for det:
.\.venv\Scripts\python.exe -c "from dotenv import load_dotenv, find_dotenv; load_dotenv(find_dotenv()); import os; os.environ['USE_BLOB_INDEXES']='true'; import azure_blob; print('container=', azure_blob.CONTAINER_NAME, 'enabled=', azure_blob.ENABLED); print('OK opplastet' if azure_blob.upload_document_store('./utils/create_lab_vectorindex/document_store.json') else 'FEILET — sjekk CONNECTION_STRING/CONTAINER_NAME')"
