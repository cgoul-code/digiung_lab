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
import logging
import os
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
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ── Config ────────────────────────────────────────────────────────────────────

INDEX_STORAGE        = os.getenv("INDEX_STORAGE",    "./blobstorage/chatbot")
INDEX_NAME           = os.getenv("INDEX_NAME",        "lab")
SIMILARITY_TOP_K     = int(os.getenv("SIMILARITY_TOP_K",   "5"))
SIMILARITY_CUTOFF    = float(os.getenv("SIMILARITY_CUTOFF", "0.3"))
DOCUMENT_STORE_PATH  = os.getenv("DOCUMENT_STORE_PATH", "./utils/create_lab_vectorindex/document_store.json")

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

# ── App setup ─────────────────────────────────────────────────────────────────

app = Quart(__name__)
app = cors(app, allow_origin="*")

_index: Optional[VectorStoreIndex] = None


async def _load_index_async():
    global _index
    persist_dir = os.path.join(INDEX_STORAGE, INDEX_NAME)
    if not os.path.isdir(persist_dir):
        logging.error("Index not found at %s. Run ingest_pdfs.py first.", persist_dir)
        return

    def _load():
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage_context)

    loop = asyncio.get_event_loop()
    _index = await loop.run_in_executor(None, _load)
    logging.info("Index loaded from %s", persist_dir)


@app.before_serving
async def _spawn_loader_after_bind():
    asyncio.create_task(_load_index_async())
    logging.info("Scheduled index-loader via create_task; server is live.")


# ── Filter builder ────────────────────────────────────────────────────────────

def _build_filters(filter_dict: dict) -> Optional[MetadataFilters]:
    """
    Build a LlamaIndex MetadataFilters object from a dict like:
        {
            "filename": "rapport.pdf",
            "segment":  "Kriminalitet",
            "publisert_arstall": 2023
        }

    All conditions are combined with AND.
    Unknown fields are ignored with a warning.
    """
    if not filter_dict:
        return None

    conditions = []
    for key, value in filter_dict.items():
        if key not in FILTERABLE_FIELDS:
            logging.warning("Ignoring unknown filter field: %s", key)
            continue
        conditions.append(
            MetadataFilter(
                key=key,
                value=value,
                operator=FilterOperator.EQ,
            )
        )

    if not conditions:
        return None

    return MetadataFilters(filters=conditions, condition=FilterCondition.AND)


# ── Document store endpoint ──────────────────────────────────────────────────

def _unique_sorted(entries, field):
    """Return sorted unique non-empty values for a field, splitting on ';'."""
    vals = set()
    for e in entries:
        raw = (e.get(field) or "").strip()
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
    body = body or {}

    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    top_k   = int(body.get("top_k",  SIMILARITY_TOP_K))
    cutoff  = float(body.get("cutoff", SIMILARITY_CUTOFF))
    filters = _build_filters(body.get("filters") or {})

    if filters:
        logging.info("Applying filters: %s", body.get("filters"))

    # ── Retrieve + synthesize in executor (LlamaIndex is sync) ───────────────
    def _run_query():
        retriever = _index.as_retriever(
            similarity_top_k=top_k,
            similarity_cutoff=cutoff,
            filters=filters,
        )
        query_engine = _index.as_query_engine(
            similarity_top_k=top_k,
            similarity_cutoff=cutoff,
            filters=filters,
        )
        nodes_with_scores = retriever.retrieve(question)
        response = query_engine.query(question)
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
