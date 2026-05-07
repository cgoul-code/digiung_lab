"""
aggregate_workflow.py
=====================
LangGraph workflow for aggregating findings across all documents in the index.

Supports multiple query_type values that change extraction + aggregation strategy:
  - "problems"  → list of problems young people face (default)
  - "moments"   → critical moments / turning points
  - "personas"  → synthesize N personas from patterns across documents
  - "free"      → open-ended, no predefined output shape

Graph:
  START
    → load_documents        # read + pre-filter document list
    → extract_per_document  # per-doc retrieval + LLM extraction
    → aggregate_findings    # merge, deduplicate, structure output
  END
"""

import json
import logging
import os
from typing import Any, Optional

from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from langgraph.graph import StateGraph, START, END


# ── Query type registry ───────────────────────────────────────────────────────
# Each entry defines:
#   extract_system  — system prompt for per-document extraction
#   extract_prompt  — user prompt template (use {question}, {tittel}, {context})
#   aggregate_system — system prompt for the final aggregation call
#   aggregate_prompt — user prompt template (use {question}, {n_docs}, {all_findings})
#   output_key      — top-level key in the JSON response

QUERY_TYPES = {
    "problems": {
        "extract_system": """Du er en faglig assistent som analyserer forskningsrapporter om barn og unge i Norge.
Trekk ut konkrete problemer og utfordringer unge mennesker møter, basert på dokumentet du får.
Svar KUN med en punktliste på norsk. Hvert punkt = ett konkret problem.
Hvis dokumentet ikke er relevant, svar: INGEN RELEVANTE FUNN.""",

        "extract_prompt": """Spørsmål: {question}

Dokumenttittel: {tittel}
{context}

List opp konkrete problemer unge møter i dette dokumentet.""",

        "aggregate_system": """Du er faglig analytiker. Slå sammen funn fra mange rapporter til en strukturert liste.
Fjern duplikater, grupper etter tema, og oppgi hvilke rapporter som støtter hvert funn.
Svar KUN i dette JSON-formatet:
{{"items": [{{"label": "Kort problemnavn", "description": "1-2 setninger", "sources": ["Tittel1"]}}]}}""",

        "aggregate_prompt": """Spørsmål: {question}
Funn fra {n_docs} rapporter:
{all_findings}
Aggreger og strukturer som beskrevet.""",

        "output_key": "problems",
    },

    "moments": {
        "extract_system": """Du er en faglig assistent. Din oppgave er å identifisere kritiske øyeblikk eller vendepunkter
i unge menneskers liv som beskrives i forskningsdokumentet du får.
Et kritisk øyeblikk er en situasjon, hendelse eller livsfase som har stor innvirkning på den unges utvikling eller helse.
Svar KUN med en punktliste på norsk. Hvert punkt = ett kritisk øyeblikk eller vendepunkt.
Hvis dokumentet ikke er relevant, svar: INGEN RELEVANTE FUNN.""",

        "extract_prompt": """Spørsmål: {question}

Dokumenttittel: {tittel}
{context}

Hvilke kritiske øyeblikk eller vendepunkter for unge beskrives i dette dokumentet?""",

        "aggregate_system": """Du er faglig analytiker. Slå sammen kritiske øyeblikk fra mange rapporter.
Grupper etter livsfase eller type hendelse. Fjern overlapp. Oppgi kildereferanser.
Svar KUN i dette JSON-formatet:
{{"items": [{{"label": "Navn på øyeblikket", "description": "Beskrivelse av situasjonen og dens innvirkning", "sources": ["Tittel1"]}}]}}""",

        "aggregate_prompt": """Spørsmål: {question}
Kritiske øyeblikk fra {n_docs} rapporter:
{all_findings}
Aggreger og strukturer som beskrevet.""",

        "output_key": "moments",
    },

    "personas": {
        "extract_system": """Du er en UX-forsker og faglig analytiker. Fra forskningsdokumentet du får skal du trekke ut
mønstre som beskriver ulike typer unge mennesker — deres situasjon, atferd, bekymringer og behov.
Svar KUN med en punktliste der hvert punkt beskriver ett mønster eller én type ung person.
Inkluder: hvem de er, hva de sliter med, og hva de trenger.
Hvis dokumentet ikke er relevant, svar: INGEN RELEVANTE FUNN.""",

        "extract_prompt": """Spørsmål: {question}

Dokumenttittel: {tittel}
{context}

Hvilke mønstre eller typer unge mennesker beskrives i dette dokumentet?""",

        "aggregate_system": """Du er en erfaren UX-forsker. Basert på mønstre fra mange rapporter skal du lage personas.
Syntesiser {n_personas} realistiske personas som representerer ulike grupper unge med ulike utfordringer.
Svar KUN i dette JSON-formatet:
{{"items": [
  {{
    "label": "Personaens navn og alder (fiktivt)",
    "description": "2-3 setninger om hvem de er og livssituasjon",
    "challenges": ["Utfordring 1", "Utfordring 2"],
    "needs": ["Behov 1", "Behov 2"],
    "sources": ["Rapport som støtter denne personaen"]
  }}
]}}""",

        "aggregate_prompt": """Spørsmål: {question}
Mønstre fra {n_docs} rapporter:
{all_findings}
Lag {n_personas} personas basert på disse mønstrene.""",

        "output_key": "personas",
    },

    "free": {
        "extract_system": """Du er en faglig assistent som analyserer forskningsrapporter om barn og unge i Norge.
Svar på spørsmålet du får, basert på innholdet i dokumentet.
Svar med en kort punktliste på norsk med de viktigste funnene relatert til spørsmålet.
Hvis dokumentet ikke er relevant, svar: INGEN RELEVANTE FUNN.""",

        "extract_prompt": """Spørsmål: {question}

Dokumenttittel: {tittel}
{context}

Hva sier dette dokumentet om spørsmålet?""",

        "aggregate_system": """Du er faglig analytiker. Gi et strukturert, helhetlig svar på spørsmålet
basert på funn fra mange rapporter. Grupper logisk og oppgi kildereferanser.
Svar KUN i dette JSON-formatet:
{{"items": [{{"label": "Tema eller poeng", "description": "Utdypende tekst", "sources": ["Tittel1"]}}]}}""",

        "aggregate_prompt": """Spørsmål: {question}
Funn fra {n_docs} rapporter:
{all_findings}
Gi et strukturert svar på spørsmålet.""",

        "output_key": "findings",
    },
}


# ── State ─────────────────────────────────────────────────────────────────────

class ChunkRef(BaseModel):
    page:    Optional[int] = None
    excerpt: str           = ""

class DocFindings(BaseModel):
    tittel:   str
    filename: str
    findings: list[str]      = Field(default_factory=list)
    chunks:   list[ChunkRef] = Field(default_factory=list)

class AggregateState(TypedDict):
    question: str
    query_type: str                     # "problems" | "moments" | "personas" | "free"
    n_personas: int                     # only used for query_type="personas"
    document_store_path: str
    index_name: str                     # key into document_store.json when format is a dict
    index: Any
    llm: Any           # kept for backward compat — used as fallback
    extract_llm: Any   # stronger model for per-document extraction
    aggregate_llm: Any # faster/cheaper model for final aggregation
    chunks_per_doc: int
    filters: dict
    documents: list[dict]
    per_doc_findings: list[DocFindings]
    result: Optional[dict]
    event_queue: Optional[Any]          # queue.Queue for SSE progress events


# ── Node: load_documents ──────────────────────────────────────────────────────

def _emit(state: AggregateState, event: dict):
    """Push an event to the SSE queue if one is present."""
    q = state.get("event_queue")
    if q is not None:
        try:
            q.put_nowait(event)
        except Exception:
            pass


def _read_doc_store_entries(path: str, index_name: str) -> list[dict]:
    """Load entries from document_store.json.
    Supports both the legacy flat-list format and the new dict-of-lists format:
      {"IndexName": [...], "OtherIndex": [...]}
    Falls back to returning all entries when index_name is not found.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if index_name and index_name in data:
        return data[index_name]
    # index not found — return all entries across all indexes as fallback
    all_entries = []
    for entries in data.values():
        if isinstance(entries, list):
            all_entries.extend(entries)
    return all_entries


def load_documents(state: AggregateState) -> dict:
    path       = state["document_store_path"]
    index_name = state.get("index_name", "")
    entries    = _read_doc_store_entries(path, index_name)
        

    filters = state.get("filters") or {}
    print(f"[load_documents] Loaded {len(entries)} documents from store. Applying filters: {filters}", flush=True)
    if filters:
        def matches(entry):
            for key, value in filters.items():
                entry_val = str(entry.get(key, "") or "").lower()
                # Value may be comma-joined multi-select (e.g. "Tittel1,Tittel2")
                # The entry matches if ANY of the selected values matches
                selected = [v.strip().lower() for v in str(value).split(";") if v.strip()]
                if not any(sel == entry_val or sel in entry_val for sel in selected):
                    return False
            return True
        entries = [e for e in entries if matches(e)]
    
    print(f"[load_documents] {len(entries)} documents after filters: {filters}", flush=True)

    logging.info("[aggregate] %d documents to visit", len(entries))
    _emit(state, {
        "event":      "node",
        "node":       "load_documents",
        "message":    f"{len(entries)} dokumenter lastet",
        "total_docs": len(entries),
    })
    return {"documents": entries}


# ── Node: extract_per_document ────────────────────────────────────────────────

def extract_per_document(state: AggregateState) -> dict:
    index: VectorStoreIndex = state["index"]
    llm = state.get("extract_llm") or state["llm"]
    print(f"[extract] Using LLM: {type(llm).__name__}", flush=True)
    question = state["question"]
    chunks_per_doc = state.get("chunks_per_doc", 4)
    query_type = state.get("query_type", "problems")
    cfg = QUERY_TYPES.get(query_type, QUERY_TYPES["free"])

    per_doc_findings: list[DocFindings] = []
    total_docs = len(state["documents"])

    _emit(state, {
        "event":      "node",
        "node":       "extract_per_document",
        "message":    f"Starter utvinning fra {total_docs} dokumenter…",
        "total_docs": total_docs,
    })

    for doc_idx, entry in enumerate(state["documents"]):
        if entry.get("url"):
            # URL-ingested entries store the URL as `filename` in chunk metadata
            filename = entry["url"]
        else:
            filename = os.path.basename(entry.get("filnavn", "").replace("\\", os.sep))
        tittel = entry.get("tittel") or filename

        _emit(state, {
            "event":    "doc_start",
            "index":    doc_idx,
            "total":    total_docs,
            "tittel":   tittel,
            "filename": filename,
        })

        doc_filter = MetadataFilters(filters=[
            MetadataFilter(key="filename", value=filename, operator=FilterOperator.EQ)
        ])
        try:
            retriever = index.as_retriever(
                similarity_top_k=chunks_per_doc,
                filters=doc_filter,
            )
            nodes = retriever.retrieve(question)
            print(f"[{doc_idx+1}/{total_docs}] {tittel[:60]!r} → {len(nodes)} chunk(s) retrieved", flush=True)
        except Exception as e:
            print(f"[{doc_idx+1}/{total_docs}] RETRIEVAL ERROR for {filename!r}: {e}", flush=True)
            logging.warning("[aggregate] Retrieval failed for: %s", filename, exc_info=True)
            nodes = []

        if not nodes:
            print(f"  ↳ No chunks found — skipping", flush=True)
            continue

        chunks = []
        context_parts = []
        for n in nodes:
            node = getattr(n, "node", n)
            meta = getattr(node, "metadata", {}) or {}
            text = (getattr(node, "text", "") or "").strip()
            raw_page = meta.get("page_label") or meta.get("page")
            chunks.append(ChunkRef(
                page=int(raw_page) if raw_page is not None else None,
                excerpt=text[:600],
            ))
            context_parts.append(text[:800])
        context = "\n\n".join(context_parts).strip()

        if not context:
            print(f"  ↳ Chunks were empty — skipping", flush=True)
            continue

        print(f"  ↳ Context length: {len(context)} chars — calling LLM…", flush=True)

        prompt = cfg["extract_prompt"].format(
            question=question,
            tittel=tittel,
            context=context,
        )

        try:
            response = llm.invoke([
                SystemMessage(content=cfg["extract_system"]),
                HumanMessage(content=prompt),
            ])
            raw = (response.content or "").strip()
            print(f"  ↳ LLM response ({len(raw)} chars): {raw[:120]!r}", flush=True)
        except Exception as e:
            print(f"  ↳ LLM ERROR: {e}", flush=True)
            logging.warning("[aggregate] LLM failed for: %s", filename, exc_info=True)
            continue

        if "INGEN RELEVANTE FUNN" in raw.upper():
            print(f"  ↳ No relevant findings", flush=True)
            continue

        findings = [
            line.lstrip("-•* ").strip()
            for line in raw.splitlines()
            if line.strip() and line.strip()[0] in "-•*"
        ]
        if not findings:
            print(f"  ↳ No bullet points found, using all lines as fallback", flush=True)
            findings = [l.strip() for l in raw.splitlines() if l.strip()]

        n_findings = len(findings) if findings else 0
        print(f"  ↳ {n_findings} finding(s) extracted", flush=True)
        _emit(state, {
            "event":      "doc_done",
            "index":      doc_idx,
            "total":      total_docs,
            "tittel":     tittel,
            "filename":   filename,
            "n_findings": n_findings,
        })

        if findings:
            per_doc_findings.append(DocFindings(
                tittel=tittel,
                filename=filename,
                findings=findings,
                chunks=chunks,
            ))
            logging.info("[aggregate] %d findings from: %s", len(findings), tittel)

    logging.info("[aggregate] %d/%d docs had findings",
                 len(per_doc_findings), len(state["documents"]))
    print(f"\n[aggregate] DONE: {len(per_doc_findings)}/{len(state['documents'])} docs had findings", flush=True)
    return {"per_doc_findings": per_doc_findings}


# ── Node: aggregate_findings ──────────────────────────────────────────────────

def aggregate_findings(state: AggregateState) -> dict:
    llm = state.get("aggregate_llm") or state["llm"]
    per_doc = state["per_doc_findings"]
    question = state["question"]
    query_type = state.get("query_type", "problems")
    n_personas = state.get("n_personas", 3)
    cfg = QUERY_TYPES.get(query_type, QUERY_TYPES["free"])

    _emit(state, {
        "event":   "node",
        "node":    "aggregate_findings",
        "message": f"Aggregerer funn fra {len(per_doc)} dokumenter…",
    })

    if not per_doc:
        return {"result": {
            "question": question,
            "query_type": query_type,
            "documents_visited": len(state["documents"]),
            "documents_with_findings": 0,
            cfg["output_key"]: [],
        }}

    all_findings_text = ""
    for doc in per_doc:
        all_findings_text += f"\n\n### {doc.tittel}\n"
        for f in doc.findings:
            all_findings_text += f"- {f}\n"

    prompt = cfg["aggregate_prompt"].format(
        question=question,
        n_docs=len(per_doc),
        all_findings=all_findings_text,
        n_personas=n_personas,
    )

    print(f"\n[aggregate] Calling aggregation LLM with {len(per_doc)} doc findings…", flush=True)
    try:
        response = llm.invoke([
            SystemMessage(content=cfg["aggregate_system"].format(n_personas=n_personas)),
            HumanMessage(content=prompt),
        ])
        raw = (response.content or "").strip()
        print(f"[aggregate] Aggregation LLM response ({len(raw)} chars): {raw[:200]!r}", flush=True)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        parsed = json.loads(raw)
        items = parsed.get("items", [])
        print(f"[aggregate] Parsed {len(items)} items from aggregation", flush=True)
    except Exception as e:
        print(f"[aggregate] AGGREGATION ERROR: {e}", flush=True)
        logging.error("[aggregate] Aggregation failed", exc_info=True)
        items = [
            {"label": f[:80], "description": f, "sources": [doc.tittel]}
            for doc in per_doc for f in doc.findings
        ]

    # Build title → sorted unique page numbers from retrieved chunks
    pages_by_title: dict[str, list[int]] = {}
    for doc in per_doc:
        pages = sorted({c.page for c in doc.chunks if c.page is not None})
        if pages:
            pages_by_title[doc.tittel] = pages

    def _enrich_sources(sources: list) -> list:
        enriched = []
        for src in sources:
            pages = pages_by_title.get(src)
            if pages:
                enriched.append(f"{src} (s. {', '.join(str(p) for p in pages)})")
            else:
                enriched.append(src)
        return enriched

    for item in items:
        if "sources" in item:
            item["sources"] = _enrich_sources(item["sources"])

    return {"result": {
        "question": question,
        "query_type": query_type,
        "documents_visited": len(state["documents"]),
        "documents_with_findings": len(per_doc),
        cfg["output_key"]: items,
    }}


# ── Build graph ───────────────────────────────────────────────────────────────

def build_aggregate_graph():
    builder = StateGraph(AggregateState)
    builder.add_node("load_documents",       load_documents)
    builder.add_node("extract_per_document", extract_per_document)
    builder.add_node("aggregate_findings",   aggregate_findings)
    builder.add_edge(START,                  "load_documents")
    builder.add_edge("load_documents",       "extract_per_document")
    builder.add_edge("extract_per_document", "aggregate_findings")
    builder.add_edge("aggregate_findings",   END)
    return builder.compile()


aggregate_graph = build_aggregate_graph()
logging.info("aggregate_graph compiled.")