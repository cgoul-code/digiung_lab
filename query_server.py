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
import json
import logging
import os
import uuid
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
from aggregate_workflow import aggregate_graph

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

INDEX_STORAGE        = _PREFIX + os.getenv("INDEX_STORAGE",    "/blobstorage/chatbot")
INDEX_NAME           = os.getenv("INDEX_NAME",        "DigiUng_lab")
SIMILARITY_TOP_K     = int(os.getenv("SIMILARITY_TOP_K",   "5"))
SIMILARITY_CUTOFF    = float(os.getenv("SIMILARITY_CUTOFF", "0.3"))
DOCUMENT_STORE_PATH  = _PREFIX + os.getenv("DOCUMENT_STORE_PATH", "/utils/create_lab_vectorindex/document_store.json")

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

_index: Optional[VectorStoreIndex] = None

# ── Job store for async aggregate jobs ───────────────────────────────────────
# job_id -> {"status": "running"|"done"|"error", "events": [...], "result": {...}}
_jobs: dict = {}


async def _load_index_async():
    global _index
    persist_dir = os.path.join(INDEX_STORAGE, INDEX_NAME)
    print(f"[index] Loading from {persist_dir} ...", flush=True)

    if not os.path.isdir(persist_dir):
        print(f"[index] ERROR: directory not found: {persist_dir}", flush=True)
        print(f"[index] Run ingest_pdfs.py first.", flush=True)
        return

    def _load():
        print(f"[index] Reading storage context ...", flush=True)
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        print(f"[index] Building index ...", flush=True)
        idx = load_index_from_storage(storage_context)
        print(f"[index] Index ready.", flush=True)
        return idx

    loop = asyncio.get_event_loop()
    try:
        _index = await loop.run_in_executor(None, _load)
        print(f"[index] Loaded successfully from {persist_dir}", flush=True)
    except Exception as e:
        print(f"[index] FAILED to load: {e}", flush=True)


@app.before_serving
async def _spawn_loader_after_bind():
    # Must return quickly — Hypercorn has a startup timeout.
    # Schedule the index load as a background task and return immediately.
    loop = asyncio.get_event_loop()
    loop.create_task(_load_index_async())
    print("[server] Server is live. Index loading in background...", flush=True)


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
        values = [v.strip() for v in str(value).split(",") if v.strip()]
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
    Returns unique dropdown options derived from document_store.json.
    The client calls this once on startup to populate its filter dropdowns.
    """
    if not os.path.isfile(DOCUMENT_STORE_PATH):
        return jsonify({"error": f"document_store.json not found at {DOCUMENT_STORE_PATH}"}), 404

    def _load():
        import json
        with open(DOCUMENT_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    loop = asyncio.get_event_loop()
    try:
        entries = await loop.run_in_executor(None, _load)
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
    return jsonify({"status": "ok", "index_loaded": _index is not None})


@app.get("/index/info")
async def index_info():
    if _index is None:
        return jsonify({"error": "Index not loaded yet"}), 503
    try:
        doc_count = len(_index.docstore.docs)
    except Exception:
        doc_count = -1
    return jsonify({
        "index_name":      INDEX_NAME,
        "storage":         INDEX_STORAGE,
        "document_chunks": doc_count,
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
    if _index is None:
        return jsonify({"error": "Index not loaded yet"}), 503

    body = await request.get_json(force=True)
    print(f"[/query] Received body: {body}", flush=True)
    body = body or {}

    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    top_k   = int(body.get("top_k",  SIMILARITY_TOP_K))
    cutoff  = float(body.get("cutoff", SIMILARITY_CUTOFF))
    params = _parse_aggregate_body(body)
    print(f"[/query] filters: {params.get('filters')}", flush=True)
    filters = _build_filters(body.get("filters") or {})
    print(f"[/query] built filters: {filters}", flush=True)

    if filters:
        logging.info("Applying filters: %s", body.get("filters"))

    # ── Retrieve + synthesize in executor (LlamaIndex is sync) ───────────────
    def _run_query():
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.response_synthesizers import get_response_synthesizer

        from llama_index.core.response_synthesizers import get_response_synthesizer

        # Retrieve using LlamaIndex filters (supports OR natively)
        retriever = _index.as_retriever(
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
    }



@app.post("/aggregate/stream")
async def aggregate_stream():
    """
    Start an aggregate job and return a job_id.
    The client then polls GET /aggregate/stream/<job_id> for progress events.
    """
    if _index is None:
        return jsonify({"error": "Index not loaded yet"}), 503

    raw = await request.get_data(as_text=True)
    print(f"[/aggregate/stream] raw body: {raw[:500]}", flush=True)
    try:
        body = json.loads(raw) if raw else {}
    except Exception as e:
        print(f"[/aggregate/stream] JSON parse error: {e}", flush=True)
        body = {}
    print(f"[/aggregate/stream] parsed body: {body}", flush=True)
    params = _parse_aggregate_body(body)
    print(f"[/aggregate/stream] filters: {params.get('filters')}", flush=True)
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
                "document_store_path": DOCUMENT_STORE_PATH,
                "index": _index, "llm": _langchain_llm,
                "documents": [], "per_doc_findings": [], "result": None,
                "event_queue": QueueProxy(),
            }
            final = aggregate_graph.invoke(state)
            loop.call_soon_threadsafe(_emit_job, {"event": "result", **(final["result"] or {})})
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

    return jsonify({
        "status": job["status"],
        "events": new_events,
        "total":  len(job["events"]),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)