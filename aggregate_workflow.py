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

        "default_question": "Hvilke problemer og utfordringer møter barn og unge?",
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

        "default_question": "Hvilke kritiske øyeblikk og vendepunkter opplever unge?",
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

        "default_question": "Hvem er de unge, hva sliter de med og hva trenger de?",
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

        "default_question": "Gi en helhetlig oppsummering av de viktigste funnene i dokumentet.",
        "output_key": "findings",
    },

    # Strategisk risiko for Helsedirektoratet (Plan & styring).
    # "structured": True → extract/aggregate produce JSON objects (not bullet lists)
    # following the analysekjeden: kildefunn → driver → sårbarhet → konsekvens → risiko.
    "strategisk_risiko": {
        "structured": True,

        "extract_system": """Du er en analyseassistent for strategisk risikoanalyse i Helsedirektoratet.
Du analyserer ett kildedokument om gangen (årsrapporter, tildelingsbrev, hovedinstruks, strategi, riksrevisjonsrapport o.l.) og utleder mulige strategiske risikoer.

Begrepsapparat du SKAL holde adskilt:
- Driver: et eksternt utviklingstrekk, styringskrav eller rammevilkår som kan påvirke direktoratets oppdrag, handlingsrom, prioriteringer eller måloppnåelse over tid. En driver er IKKE en risiko i seg selv.
- Sårbarhet: et forhold ved direktoratets ansvar, rolle, kapasitet, kompetanse, styring, samhandling, data, teknologi, regelverksetterlevelse eller avhengigheter som kan svekke evnen til å møte en driver.
- Konsekvens: hva det kan bety for måloppnåelse, samfunnsoppdrag, ressursbruk, styring og kontroll, sikkerhet, beredskap, legitimitet eller tillit.
- Risiko: en usikkerhet som kan påvirke direktoratets evne til å ivareta samfunnsoppdrag, måloppnåelse, styringskrav eller prioriteringsevne over 3-5 år. Risiko oppstår når en driver møter en sårbarhet og kan gi vesentlig konsekvens.

Arbeid etter analysekjeden: kildefunn -> driver -> relevans -> sårbarhet -> konsekvens -> risiko -> avklaringsspørsmål.
Bruk et nøkternt, presist og direktoratstilpasset språk. Unngå konsulentspråk, dramatisering og bastante konklusjoner. Ikke foreslå tiltak. Ikke forveksle drivere, sårbarheter, konsekvenser og risiko. Ikke gjør operative forhold strategiske uten å forklare hvorfor de har strategisk betydning.

Svar KUN med gyldig JSON i dette formatet (ingen tekst utenfor JSON):
{
  "relevant": true,
  "relevans": "kort vurdering av dokumentets relevans for strategisk risiko",
  "kildefunn": ["det dokumentet faktisk sier, direkte forankret i kilden"],
  "drivere": ["mulige strategiske drivere"],
  "sarbarheter": ["mulige sårbarheter som bør undersøkes, formulert som hypoteser/spørsmål"],
  "konsekvenser": ["mulige konsekvenser"],
  "risikoer": ["foreløpige strategiske risikoer, formulert som usikkerhet over 3-5 år"],
  "avklaringssporsmal": ["spørsmål til videre avklaring"],
  "kildegrunnlag_styrke": "kort vurdering av hvor sterkt kildegrunnlaget er"
}
Hvis dokumentet ikke er relevant for strategisk risiko, svar: {"relevant": false}.""",

        "extract_prompt": """Spørsmål/fokus: {question}

Dokumenttittel: {tittel}
{context}

Analyser dette dokumentet etter analysekjeden og svar med JSON som beskrevet.""",

        "aggregate_system": """Du er analytiker i risikoteamet. Du får analyser per dokument og skal lage en syntese på tvers - en longlist over mulige strategiske risikoområder.
Identifiser mønstre, slå sammen overlappende funn, og hold driver/sårbarhet/konsekvens/risiko adskilt. Oppgi hvilke kilder (titler) som peker i samme retning.
Bruk et nøkternt, presist og direktoratstilpasset språk. Ikke foreslå tiltak.

Svar KUN med gyldig JSON i dette formatet (ingen tekst utenfor JSON):
{{
  "monstre": ["overordnede mønstre på tvers av dokumentene"],
  "temaer": [
    {{
      "label": "Kort navn på risikoområde/tema",
      "beskrivelse": "1-3 setninger",
      "drivere": ["drivere som støtter temaet"],
      "sarbarheter": ["mulige sårbarheter som bør undersøkes"],
      "konsekvenser": ["mulige konsekvenser"],
      "risikoer": ["foreløpige strategiske risikoer"],
      "sources": ["Tittel1", "Tittel2"]
    }}
  ],
  "usikkerhet_kunnskapshull": ["usikkerhet og kunnskapshull"],
  "sporsmal_til_ledergruppen": ["spørsmål til ledergruppen"]
}}""",

        "aggregate_prompt": """Spørsmål/fokus: {question}
Analyser per dokument fra {n_docs} dokumenter:
{all_findings}
Lag en syntese på tvers (longlist) som beskrevet.""",

        "default_question": "Hvilke strategiske drivere, sårbarheter, konsekvenser og risikoer fremgår av dokumentet?",
        "output_key": "risikoomrader",
    },
}


# ── State ─────────────────────────────────────────────────────────────────────

class ChunkRef(BaseModel):
    page:    Optional[int] = None
    excerpt: str           = ""

class DocFindings(BaseModel):
    tittel:             str
    filename:           str
    kilde_url:          str            = ""
    publisert_av:       str            = ""
    publisert_arstall:  Optional[int]  = None
    findings:           list[str]      = Field(default_factory=list)
    structured:         Optional[dict] = None   # set for "structured" query types
    chunks:             list[ChunkRef] = Field(default_factory=list)

class AggregateState(TypedDict):
    question: str
    query_type: str                     # "problems" | "moments" | "personas" | "free" | "strategisk_risiko"
    query_type_cfg: Any                 # effective (possibly user-edited) prompt config; falls back to QUERY_TYPES
    n_personas: int                     # only used for query_type="personas"
    include_aggregate: bool             # run cross-document syntese (default True)
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
    cancel_event: Optional[Any]         # threading.Event set when user cancels


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


def _parse_json_block(raw: str) -> dict:
    """Parse a JSON object from an LLM response, tolerating ```json fences."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# Generic fallback when neither a user question nor a query-type default exists.
_GENERIC_QUESTION = "Gi en helhetlig analyse av dokumentet."


def _coerce_page(raw) -> Optional[int]:
    """PDF page labels can be non-numeric (e.g. Roman numerals 'iv' for front
    matter). Return the page as an int when numeric, otherwise None — so a
    quirky label never crashes extraction."""
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _resolve_question(state: AggregateState, cfg: dict) -> str:
    """The question used for retrieval and {question} substitution.

    When the user leaves the question empty, the analysis is driven purely by the
    system prompt: we fall back to the query type's `default_question` (then a
    generic phrase) so per-document retrieval still surfaces relevant chunks.
    """
    q = (state.get("question") or "").strip()
    if q:
        return q
    return (cfg.get("default_question") or "").strip() or _GENERIC_QUESTION


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
    chunks_per_doc = state.get("chunks_per_doc", 4)
    query_type = state.get("query_type", "problems")
    cfg = state.get("query_type_cfg") or QUERY_TYPES.get(query_type, QUERY_TYPES["free"])
    # Empty question → drive the analysis from the system prompt alone.
    question = _resolve_question(state, cfg)

    per_doc_findings: list[DocFindings] = []
    total_docs = len(state["documents"])

    _emit(state, {
        "event":      "node",
        "node":       "extract_per_document",
        "message":    f"Starter utvinning fra {total_docs} dokumenter…",
        "total_docs": total_docs,
    })

    cancel_event = state.get("cancel_event")
    for doc_idx, entry in enumerate(state["documents"]):
        if cancel_event is not None and cancel_event.is_set():
            logging.info("[aggregate] Cancellation requested — stopping after %d/%d docs",
                         doc_idx, total_docs)
            _emit(state, {
                "event":   "cancelled",
                "message": f"Avbrutt etter {doc_idx}/{total_docs} dokumenter",
                "index":   doc_idx,
                "total":   total_docs,
            })
            break
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
                page=_coerce_page(raw_page),
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

        structured = None
        if cfg.get("structured"):
            try:
                structured = _parse_json_block(raw)
            except Exception as e:
                print(f"  ↳ JSON parse error: {e}", flush=True)
                logging.warning("[aggregate] structured parse failed for %s", filename, exc_info=True)
                continue
            if not structured or structured.get("relevant") is False:
                print(f"  ↳ Not relevant — skipping", flush=True)
                continue
            # Flat list (risks first, else kildefunn) for backward-compatible summaries.
            findings = list(structured.get("risikoer") or []) or list(structured.get("kildefunn") or [])
            has_content = any(
                structured.get(k)
                for k in ("kildefunn", "drivere", "sarbarheter", "konsekvenser", "risikoer")
            )
        else:
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
            has_content = bool(findings)

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

        if has_content:
            ar = entry.get("publisert_arstall")
            try:
                ar_int = int(ar) if ar not in (None, "") else None
            except (TypeError, ValueError):
                ar_int = None
            per_doc_findings.append(DocFindings(
                tittel=tittel,
                filename=filename,
                kilde_url=(entry.get("kilde_url") or ""),
                publisert_av=(entry.get("publisert_av") or ""),
                publisert_arstall=ar_int,
                findings=findings,
                structured=structured,
                chunks=chunks,
            ))
            logging.info("[aggregate] findings from: %s", tittel)

    logging.info("[aggregate] %d/%d docs had findings",
                 len(per_doc_findings), len(state["documents"]))
    print(f"\n[aggregate] DONE: {len(per_doc_findings)}/{len(state['documents'])} docs had findings", flush=True)
    return {"per_doc_findings": per_doc_findings}


# ── Node: aggregate_findings ──────────────────────────────────────────────────

def aggregate_findings(state: AggregateState) -> dict:
    llm = state.get("aggregate_llm") or state["llm"]
    per_doc = state["per_doc_findings"]
    query_type = state.get("query_type", "problems")
    n_personas = state.get("n_personas", 3)
    cfg = state.get("query_type_cfg") or QUERY_TYPES.get(query_type, QUERY_TYPES["free"])
    # Empty question → drive the analysis from the system prompt alone.
    question = _resolve_question(state, cfg)
    structured = bool(cfg.get("structured"))
    include_aggregate = state.get("include_aggregate", True)

    cancel_event = state.get("cancel_event")
    if cancel_event is not None and cancel_event.is_set():
        return {"result": {
            "question": question,
            "query_type": query_type,
            "documents_visited": len(state["documents"]),
            "documents_with_findings": len(per_doc),
            "cancelled": True,
            cfg["output_key"]: [],
        }}

    # Cross-document syntese is optional — per-document analysis is always returned
    # separately by the server. When not requested, skip the aggregation LLM call.
    if not include_aggregate:
        return {"result": {
            "question": question,
            "query_type": query_type,
            "documents_visited": len(state["documents"]),
            "documents_with_findings": len(per_doc),
            "aggregated": False,
            cfg["output_key"]: [],
        }}

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

    _RISK_KEYS = (
        ("kildefunn", "Kildefunn"), ("drivere", "Drivere"),
        ("sarbarheter", "Sårbarheter"), ("konsekvenser", "Konsekvenser"),
        ("risikoer", "Risikoer"),
    )

    all_findings_text = ""
    for doc in per_doc:
        all_findings_text += f"\n\n### {doc.tittel}\n"
        if structured and doc.structured:
            for key, lbl in _RISK_KEYS:
                vals = doc.structured.get(key) or []
                if vals:
                    all_findings_text += f"{lbl}:\n" + "".join(f"- {v}\n" for v in vals)
        else:
            for f in doc.findings:
                all_findings_text += f"- {f}\n"

    prompt = cfg["aggregate_prompt"].format(
        question=question,
        n_docs=len(per_doc),
        all_findings=all_findings_text,
        n_personas=n_personas,
    )

    print(f"\n[aggregate] Calling aggregation LLM with {len(per_doc)} doc findings…", flush=True)
    parsed: dict = {}
    try:
        response = llm.invoke([
            SystemMessage(content=cfg["aggregate_system"].format(n_personas=n_personas)),
            HumanMessage(content=prompt),
        ])
        raw = (response.content or "").strip()
        print(f"[aggregate] Aggregation LLM response ({len(raw)} chars): {raw[:200]!r}", flush=True)
        parsed = _parse_json_block(raw)
        items = parsed.get("temaer", []) if structured else parsed.get("items", [])
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
    url_by_title: dict[str, str] = {}
    for doc in per_doc:
        pages = sorted({c.page for c in doc.chunks if c.page is not None})
        if pages:
            pages_by_title[doc.tittel] = pages
        if doc.kilde_url:
            url_by_title[doc.tittel] = doc.kilde_url

    def _enrich_sources(sources: list) -> list:
        enriched = []
        for src in sources:
            tittel = src if isinstance(src, str) else (src.get("tittel") or "")
            enriched.append({
                "tittel":    tittel,
                "pages":     pages_by_title.get(tittel, []),
                "kilde_url": url_by_title.get(tittel, ""),
            })
        return enriched

    for item in items:
        if "sources" in item:
            item["sources"] = _enrich_sources(item["sources"])

    result = {
        "question": question,
        "query_type": query_type,
        "documents_visited": len(state["documents"]),
        "documents_with_findings": len(per_doc),
        "aggregated": True,
        cfg["output_key"]: items,
    }
    if structured:
        # Syntese-level fields that aren't per-tema (longlist context).
        result["monstre"] = parsed.get("monstre", [])
        result["usikkerhet_kunnskapshull"] = parsed.get("usikkerhet_kunnskapshull", [])
        result["sporsmal_til_ledergruppen"] = parsed.get("sporsmal_til_ledergruppen", [])
    return {"result": result}


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