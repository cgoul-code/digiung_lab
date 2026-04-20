"""
query_server.py
===============
Quart HTTP server that loads a persisted LlamaIndex VectorStoreIndex
and answers natural-language queries via HTTP.

Supports optional metadata filtering on any document_store field:
    filename, tittel, publisert_av, type_kilde, malgruppe, segment,
    publisert_arstall, antall_deltakere

Run:
    ./.venv/Scripts/python.exe query_server.py

Endpoints:
    GET  /health        → liveness check
    GET  /index/info    → basic stats about the loaded index
    POST /query         → ask a question, get an answer + sources
"""

import asyncio
import datetime
import json
import logging
import os
import uuid
from io import BytesIO
from typing import Optional

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from quart import Quart, request, jsonify
from quart_cors import cors

from llama_index.core import (
    StorageContext,
    Settings,
    load_index_from_storage,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)
# Note: for multi-value OR within a field we nest MetadataFilters
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from aggregate_workflow import aggregate_graph, QUERY_TYPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ── Environment detection ─────────────────────────────────────────────────────

def running_locally() -> bool:
    if 'WEBSITE_SITE_NAME' in os.environ or 'FUNCTIONS_WORKER_RUNTIME' in os.environ:
        return False
    print("Running locally", flush=True)
    return True

_LOCAL = running_locally()
_PREFIX = "." if _LOCAL else ""

# ── Config ────────────────────────────────────────────────────────────────────

INDEX_STORAGE     = _PREFIX + os.getenv("INDEX_STORAGE", "/blobstorage/chatbot")
SIMILARITY_TOP_K  = int(os.getenv("SIMILARITY_TOP_K",   "5"))
SIMILARITY_CUTOFF = float(os.getenv("SIMILARITY_CUTOFF", "0.3"))

# Fallback document store used when an index has no local document_store.json
_doc_store_env = os.getenv("DOCUMENT_STORE_PATH")
DOCUMENT_STORE_PATH = _doc_store_env or "./utils/create_lab_vectorindex/document_store.json"

print(f"[config] LOCAL={_LOCAL}", flush=True)
print(f"[config] INDEX_STORAGE={INDEX_STORAGE}", flush=True)
print(f"[config] DOCUMENT_STORE_PATH (fallback)={DOCUMENT_STORE_PATH}", flush=True)

# All metadata fields that can be filtered on
FILTERABLE_FIELDS = {
    "filename",
    "tittel",
    "publisert_av",
    "type_kilde",
    "malgruppe",
    "segment",
    "publisert_arstall",
    "antall_deltakere",
}

# ── LlamaIndex models ─────────────────────────────────────────────────────────

Settings.embed_model = AzureOpenAIEmbedding(
    model=os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL"),
    deployment_name=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDINGS_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION"),
)

Settings.llm = AzureOpenAI(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0,
)

Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=128)

# LangChain LLM for LangGraph workflows (aggregate_workflow uses this)
_langchain_llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0.0,
    timeout=120,
)

# ── App setup ─────────────────────────────────────────────────────────────────

app = Quart(__name__)
app = cors(app, allow_origin="*")

# name -> {"index": VectorStoreIndex, "doc_store_path": str}
_indexes: dict[str, dict] = {}


def _read_doc_store_entries(doc_store_path: str, index_name: str) -> list[dict]:
    """Load entries for a given index from document_store.json.
    Supports legacy flat-list format and new dict-of-lists format:
      {"IndexName": [...], "OtherIndex": [...]}
    """
    with open(doc_store_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if index_name and index_name in data:
        return data[index_name]
    all_entries: list[dict] = []
    for v in data.values():
        if isinstance(v, list):
            all_entries.extend(v)
    return all_entries

# ── Job store for async aggregate jobs ───────────────────────────────────────
# job_id -> {"status": "running"|"done"|"error", "events": [...], "result": {...}}
_jobs: dict = {}


def _resolve_index(name: Optional[str]):
    """Return (index_name, entry) for the requested name, or the first loaded index as fallback."""
    if name and name in _indexes:
        return name, _indexes[name]
    first = next(iter(_indexes), None)
    return first, _indexes.get(first)


async def _load_all_indexes_async():
    global _indexes
    if not os.path.isdir(INDEX_STORAGE):
        print(f"[index] INDEX_STORAGE not found: {INDEX_STORAGE}", flush=True)
        return

    # Derive the list of index names from document_store.json keys only
    if os.path.isfile(DOCUMENT_STORE_PATH):
        with open(DOCUMENT_STORE_PATH, "r", encoding="utf-8") as _f:
            _ds = json.load(_f)
        names = sorted(_ds.keys()) if isinstance(_ds, dict) else []
    else:
        names = []

    # Filter to those that actually have a built index on disk
    names = [n for n in names if os.path.isfile(os.path.join(INDEX_STORAGE, n, "docstore.json"))]

    if not names:
        print(f"[index] No matching indexes found in {DOCUMENT_STORE_PATH} / {INDEX_STORAGE}", flush=True)
        return

    loop = asyncio.get_event_loop()
    for name in names:
        persist_dir = os.path.join(INDEX_STORAGE, name)
        local_store = os.path.join(persist_dir, "document_store.json")
        doc_store_path = local_store if os.path.isfile(local_store) else DOCUMENT_STORE_PATH

        def _load(d=persist_dir, n=name):
            print(f"[index] Loading '{n}' from {d} ...", flush=True)
            ctx = StorageContext.from_defaults(persist_dir=d)
            idx = load_index_from_storage(ctx)
            print(f"[index] '{n}' ready.", flush=True)
            return idx

        try:
            idx = await loop.run_in_executor(None, _load)
            _indexes[name] = {"index": idx, "doc_store_path": doc_store_path}
        except Exception as e:
            print(f"[index] FAILED to load '{name}': {e}", flush=True)

    print(f"[index] Loaded {len(_indexes)} index(es): {list(_indexes)}", flush=True)


@app.before_serving
async def _spawn_loader_after_bind():
    loop = asyncio.get_event_loop()
    loop.create_task(_load_all_indexes_async())
    print("[server] Server is live. Indexes loading in background...", flush=True)


# ── Filter builder ────────────────────────────────────────────────────────────

def _build_filters(filter_dict: dict) -> Optional[MetadataFilters]:
    """
    Build LlamaIndex MetadataFilters.
    - Single value per field  → EQ filter
    - Multiple values (comma-joined) → OR group for that field
    - Multiple fields → AND across fields
    """
    if not filter_dict:
        return None

    and_conditions = []

    for key, value in filter_dict.items():
        if key not in FILTERABLE_FIELDS:
            logging.warning("Ignoring unknown filter field: %s", key)
            continue
        values = [v.strip() for v in str(value).split(";") if v.strip()]
        if not values:
            continue
        if len(values) == 1:
            and_conditions.append(
                MetadataFilter(key=key, value=values[0], operator=FilterOperator.EQ)
            )
        else:
            # OR group within this field
            and_conditions.append(
                MetadataFilters(
                    filters=[MetadataFilter(key=key, value=v) for v in values],
                    condition=FilterCondition.OR,
                )
            )

    if not and_conditions:
        return None

    if len(and_conditions) == 1:
        c = and_conditions[0]
        return c if isinstance(c, MetadataFilters) else MetadataFilters(filters=[c], condition=FilterCondition.AND)

    return MetadataFilters(filters=and_conditions, condition=FilterCondition.AND)


# ── Document store endpoint ──────────────────────────────────────────────────

def _unique_sorted(entries, field):
    """Return sorted unique non-empty values for a field, splitting on ';'."""
    vals = set()
    for e in entries:
        raw = str(e.get(field) or "").strip()
        for part in raw.split(";"):
            part = part.strip()
            if part:
                vals.add(part)
    return sorted(vals)

@app.get("/document-store/filter-options")
async def document_store_filter_options():
    """
    Returns unique dropdown options derived from the index's document_store.json.
    Pass ?index_name=<name> to target a specific index.
    """
    req_index, entry = _resolve_index(request.args.get("index_name"))
    doc_store_path = entry["doc_store_path"] if entry else DOCUMENT_STORE_PATH

    if not os.path.isfile(doc_store_path):
        return jsonify({"error": f"document_store.json not found at {doc_store_path}"}), 404

    loop = asyncio.get_event_loop()
    try:
        entries = await loop.run_in_executor(
            None, _read_doc_store_entries, doc_store_path, req_index or ""
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "tittel":            _unique_sorted(entries, "tittel"),
        "publisert_av":      _unique_sorted(entries, "publisert_av"),
        "segment":           _unique_sorted(entries, "segment"),
        "type_kilde":        _unique_sorted(entries, "type_kilde"),
        "malgruppe":         _unique_sorted(entries, "malgruppe"),
        "publisert_arstall": _unique_sorted(entries, "publisert_arstall"),
    })


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return jsonify({"status": "ok", "indexes_loaded": list(_indexes)})


@app.get("/indexes")
async def list_indexes():
    if not os.path.isfile(DOCUMENT_STORE_PATH):
        return jsonify([])
    def _load():
        with open(DOCUMENT_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return sorted(data.keys()) if isinstance(data, dict) else []
    loop = asyncio.get_event_loop()
    try:
        return jsonify(await loop.run_in_executor(None, _load))
    except Exception:
        return jsonify([])


@app.get("/query-types")
async def query_types():
    return jsonify(QUERY_TYPES)


@app.get("/index/info")
async def index_info():
    if not _indexes:
        return jsonify({"error": "No indexes loaded yet"}), 503
    result = {}
    for name, entry in _indexes.items():
        try:
            doc_count = len(entry["index"].docstore.docs)
        except Exception:
            doc_count = -1
        result[name] = {"document_chunks": doc_count}
    return jsonify({
        "storage":           INDEX_STORAGE,
        "indexes":           result,
        "filterable_fields": sorted(FILTERABLE_FIELDS),
    })


@app.post("/query")
async def query():
    """
    POST /query
    Body (JSON):
        {
            "question": "Hva sier rapporten om skolefravær?",
            "top_k":    5,          // optional
            "cutoff":   0.3,        // optional
            "filters":  {           // optional — all combined with AND
                "filename":          "NIFUrapport2023-14.pdf",
                "segment":           "Skolefravær",
                "publisert_av":      "NIFU",
                "publisert_arstall": 2023,
                "type_kilde":        "Rapport",
                "malgruppe":         "Barn og unge"
            }
        }
    """
    body = await request.get_json(force=True)
    print(f"[/query] Received body: {body}", flush=True)
    body = body or {}

    index_name, entry = _resolve_index(body.get("index_name"))
    if not entry:
        return jsonify({"error": "No indexes loaded yet"}), 503

    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    top_k   = int(body.get("top_k",  SIMILARITY_TOP_K))
    cutoff  = float(body.get("cutoff", SIMILARITY_CUTOFF))
    filters = _build_filters(body.get("filters") or {})
    print(f"[/query] index={index_name} filters={filters}", flush=True)

    # ── Retrieve + synthesize in executor (LlamaIndex is sync) ───────────────
    def _run_query():
        from llama_index.core.response_synthesizers import get_response_synthesizer

        retriever = entry["index"].as_retriever(
            similarity_top_k=top_k,
            similarity_cutoff=cutoff,
            filters=filters,
        )
        nodes_with_scores = retriever.retrieve(question)
        print(f"[_run_query] retrieved {len(nodes_with_scores)} nodes, filters={filters}", flush=True)

        # Synthesize answer from the filtered nodes directly
        synthesizer = get_response_synthesizer()
        response = synthesizer.synthesize(question, nodes=nodes_with_scores)

        return str(response), nodes_with_scores

    loop = asyncio.get_event_loop()
    try:
        answer, nodes_with_scores = await loop.run_in_executor(None, _run_query)
    except Exception as e:
        logging.error("Query failed: %s", e, exc_info=True)
        return jsonify({"error": "Query failed", "detail": str(e)}), 500

    # ── Build sources ─────────────────────────────────────────────────────────
    sources = []
    seen: set[str] = set()

    for nws in nodes_with_scores:
        node     = getattr(nws, "node", nws)
        meta     = getattr(node, "metadata", {}) or {}
        text     = (getattr(node, "text", None) or "").strip()
        chunk_id = getattr(node, "id_", "") or getattr(node, "node_id", "")

        if chunk_id in seen:
            continue
        seen.add(chunk_id)

        sources.append({
            "filename":         meta.get("filename", "unknown"),
            "tittel":           meta.get("tittel", ""),
            "publisert_av":     meta.get("publisert_av", ""),
            "publisert_arstall":meta.get("publisert_arstall"),
            "type_kilde":       meta.get("type_kilde", ""),
            "malgruppe":        meta.get("malgruppe", ""),
            "segment":          meta.get("segment", ""),
            "page_number":      meta.get("page_label") or meta.get("page"),
            "score":            round(float(getattr(nws, "score", 0.0)), 4),
            "excerpt":          text[:300],
        })

    return jsonify({
        "question": question,
        "filters":  body.get("filters") or {},
        "answer":   answer,
        "sources":  sources,
    })


def _parse_aggregate_body(body):
    return {
        "question":      body.get("question", "").strip(),
        "query_type":    body.get("query_type", "problems"),
        "n_personas":    int(body.get("n_personas", 3)),
        "chunks_per_doc":int(body.get("chunks_per_doc", 4)),
        "filters":       body.get("filters") or {},
        "index_name":    body.get("index_name", ""),
    }



@app.post("/aggregate/stream")
async def aggregate_stream():
    """
    Start an aggregate job and return a job_id.
    The client then polls GET /aggregate/stream/<job_id> for progress events.
    """
    raw = await request.get_data(as_text=True)
    print(f"[/aggregate/stream] raw body: {raw[:500]}", flush=True)
    try:
        body = json.loads(raw) if raw else {}
    except Exception as e:
        print(f"[/aggregate/stream] JSON parse error: {e}", flush=True)
        body = {}
    print(f"[/aggregate/stream] parsed body: {body}", flush=True)
    params = _parse_aggregate_body(body)

    index_name, entry = _resolve_index(params.get("index_name"))
    if not entry:
        return jsonify({"error": "No indexes loaded yet"}), 503

    print(f"[/aggregate/stream] index={index_name} filters={params.get('filters')}", flush=True)
    if not params["question"]:
        return jsonify({"error": "Missing 'question'"}), 400

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "events": [], "result": None}

    loop = asyncio.get_event_loop()

    def _emit_job(item):
        if item is None:
            return
        _jobs[job_id]["events"].append(item)
        if item.get("event") == "result":
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = item
        elif item.get("event") == "error":
            _jobs[job_id]["status"] = "error"

    class QueueProxy:
        def put_nowait(self, item): loop.call_soon_threadsafe(_emit_job, item)
        def put(self, item):        loop.call_soon_threadsafe(_emit_job, item)

    def _run():
        try:
            state = {**params,
                "document_store_path": entry["doc_store_path"],
                "index_name": index_name,
                "index": entry["index"], "llm": _langchain_llm,
                "documents": [], "per_doc_findings": [], "result": None,
                "event_queue": QueueProxy(),
            }
            final = aggregate_graph.invoke(state)
            per_doc = [
                {
                    "tittel":    f.tittel,
                    "filename":  f.filename,
                    "findings":  f.findings,
                    "chunks":    [{"page": c.page, "excerpt": c.excerpt} for c in f.chunks],
                }
                for f in (final.get("per_doc_findings") or [])
                if f.findings
            ]
            loop.call_soon_threadsafe(_emit_job, {
                "event":            "result",
                **(final["result"] or {}),
                "index_name":       index_name,
                "per_doc_findings": per_doc,
            })
        except Exception as e:
            logging.error("[/aggregate/stream] failed: %s", e, exc_info=True)
            loop.call_soon_threadsafe(_emit_job, {"event": "error", "message": str(e)})
        finally:
            _jobs[job_id]["status"] = _jobs[job_id].get("status", "done")

    loop.run_in_executor(None, _run)

    return jsonify({"job_id": job_id})


@app.get("/aggregate/stream/<job_id>")
async def aggregate_stream_poll(job_id):
    """
    Poll for job progress. Returns all events since last_index.
    GET /aggregate/stream/<job_id>?last=0
    """
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    last = int(request.args.get("last", 0))
    new_events = job["events"][last:]
    status = job["status"]
    total  = len(job["events"])

    # Clean up error jobs immediately; done jobs are kept until report is downloaded
    if status == "error" and last >= total:
        _jobs.pop(job_id, None)

    return jsonify({
        "status": status,
        "events": new_events,
        "total":  total,
    })


def _generate_report_docx(result: dict) -> bytes:
    from docx import Document

    OUTPUT_KEYS = {"problems": "problems", "moments": "moments", "personas": "personas", "free": "findings"}

    doc     = Document()
    qt      = result.get("query_type", "")
    idx     = result.get("index_name", "")
    items   = result.get(OUTPUT_KEYS.get(qt, "findings"), [])
    per_doc = result.get("per_doc_findings", [])

    doc.add_heading(result.get("question", "Rapport"), level=1)

    meta = doc.add_paragraph()
    meta.add_run(f"Query type: {qt}").bold = True
    if idx:
        doc.add_paragraph(f"Index: {idx}")
    doc.add_paragraph(f"Generert: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    doc.add_paragraph(
        f"{result.get('documents_visited', 0)} dokumenter besøkt · "
        f"{result.get('documents_with_findings', 0)} med funn · "
        f"{len(items)} resultater"
    )

    if items:
        doc.add_heading("Aggregerte funn", level=2)
        for item in items:
            doc.add_heading(item.get("label", ""), level=3)
            if item.get("description"):
                doc.add_paragraph(item["description"])
            for challenge in item.get("challenges") or []:
                doc.add_paragraph(challenge, style="List Bullet")
            if item.get("needs"):
                doc.add_paragraph("Behov:").runs[0].bold = True
                for need in item["needs"]:
                    doc.add_paragraph(need, style="List Bullet")
            if item.get("sources"):
                p = doc.add_paragraph("Kilder: ")
                p.add_run("; ".join(item["sources"])).italic = True

    if per_doc:
        doc.add_heading("Funn per dokument", level=2)
        for entry in per_doc:
            doc.add_heading(entry.get("tittel") or entry.get("filename", ""), level=3)
            for finding in entry.get("findings", []):
                doc.add_paragraph(finding, style="List Bullet")
            chunks = entry.get("chunks") or []
            if chunks:
                doc.add_paragraph("Kildehenvisninger:", style="Intense Quote")
                for chunk in chunks:
                    page = chunk.get("page")
                    excerpt = (chunk.get("excerpt") or "").strip()
                    label = f"[Side {page}]  " if page is not None else ""
                    p = doc.add_paragraph(style="List Bullet 2")
                    if label:
                        p.add_run(label).bold = True
                    p.add_run(excerpt)

    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


@app.get("/aggregate/report/<job_id>")
async def aggregate_report(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found — may have already been downloaded"}), 404
    if job["status"] != "done":
        return jsonify({"error": "Job not finished yet"}), 400

    result = job.get("result") or {}
    loop = asyncio.get_event_loop()
    try:
        docx_bytes = await loop.run_in_executor(None, _generate_report_docx, result)
    except Exception as e:
        logging.error("Report generation failed: %s", e, exc_info=True)
        return jsonify({"error": "Report generation failed", "detail": str(e)}), 500

    _jobs.pop(job_id, None)

    from quart import Response
    question_slug = "".join(c if c.isalnum() else "_" for c in result.get("question", "rapport")[:40])
    filename = f"{question_slug}.docx"
    return Response(
        docx_bytes,
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)