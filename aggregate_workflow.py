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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
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


# How many per-document extractions to run at once. Each is one blocking LLM
# call, so this is the main speed lever for an analysis — capped to stay under
# Azure OpenAI rate limits. Override with AGGREGATE_EXTRACT_CONCURRENCY.
EXTRACT_CONCURRENCY = max(1, int(os.getenv("AGGREGATE_EXTRACT_CONCURRENCY", "5")))


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
Svar KUN med en punktliste. Hvert punkt = ett konkret problem.
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
Svar KUN med en punktliste. Hvert punkt = ett kritisk øyeblikk eller vendepunkt.
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
        "extract_system": """Du er en faglig assistent som analyserer dokumenter.
Svar på spørsmålet du får, basert på innholdet i dokumentet.
Svar med en kort punktliste med de viktigste funnene relatert til spørsmålet.
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

        # The JSON above, restated as data. The nodes read this rather than the
        # key names, so a template built in the wizard runs the same code path.
        "doc_notes": [
            {"key": "relevans", "label": "Relevans", "lead": True},
            {"key": "kildegrunnlag_styrke", "label": "Kildegrunnlagets styrke"},
        ],
        "doc_fields": [
            {"key": "kildefunn",    "label": "Kildefunn"},
            {"key": "drivere",      "label": "Drivere"},
            {"key": "sarbarheter",  "label": "Mulige sårbarheter"},
            {"key": "konsekvenser", "label": "Mulige konsekvenser"},
            {"key": "risikoer",     "label": "Foreløpige risikoer", "tone": "danger"},
            # Questions are for the reader, not for the synthesis to merge.
            {"key": "avklaringssporsmal", "label": "Avklaringsspørsmål", "context": False},
        ],
        "agg_items_key": "temaer",
        "agg_items_label": "risikoområder",
        "agg_item_fields": [
            {"key": "drivere",      "label": "Drivere"},
            {"key": "sarbarheter",  "label": "Sårbarheter"},
            {"key": "konsekvenser", "label": "Konsekvenser"},
            {"key": "risikoer",     "label": "Risikoer", "tone": "danger"},
        ],
        "agg_top_fields": [
            {"key": "monstre", "label": "Overordnede mønstre", "lead": True},
            {"key": "usikkerhet_kunnskapshull",  "label": "Usikkerhet og kunnskapshull"},
            {"key": "sporsmal_til_ledergruppen", "label": "Spørsmål til ledergruppen"},
        ],
    },

    # WHO-kode compliance: regelverkssjekk mot WHO-koden (International Code of
    # Marketing of Breast-milk Substitutes) + WHA-resolusjoner + Baby-Friendly /
    # Mor-barn-vennlig-standarden. Non-structured: extract → kort, forankret
    # punktliste per kilde; aggregate → ett tydelig compliance-svar (items).
    "who_kode": {
        "extract_system": """Du er en fagassistent som vurderer etterlevelse av WHO-koden (International Code of Marketing of Breast-milk Substitutes) med påfølgende WHA-resolusjoner, samt Baby-Friendly / Mor-barn-vennlig-standarden.
Du analyserer ETT kildedokument om gangen og trekker ut KUN det som er relevant for spørsmålet.
For hvert relevant punkt: gjengi hva dokumentet faktisk sier, og oppgi presis forankring (artikkel/avsnitt i Koden, resolusjonsnummer/-år, eller trinn i Baby-Friendly-standarden) når det fremgår av dokumentet.
Ikke konkluder bastant på tvers av kilder her — det gjøres i syntesen. Ikke dikt opp bestemmelser som ikke står i dokumentet.
Kildene kan være på et annet språk enn svarspråket; svar likevel på det språket som er angitt til slutt.
Svar KUN med en punktliste. Hvis dokumentet ikke er relevant for spørsmålet, svar: INGEN RELEVANTE FUNN.""",

        "extract_prompt": """Spørsmål: {question}

Dokumenttittel: {tittel}
{context}

Trekk ut bestemmelser, vilkår, definisjoner og henvisninger i dette dokumentet som er relevante for spørsmålet.""",

        "aggregate_system": """Du er fagperson på WHO-koden, WHA-resolusjonene og Baby-Friendly-standarden. Du får utdrag fra flere kilder og skal gi ETT tydelig, etterprøvbart svar på spørsmålet.
Bygg svaret KUN på utdragene; ikke legg til regler som ikke er forankret i dem. Hvis kildene er utilstrekkelige eller spriker, si det eksplisitt i konklusjonen («Uklart») og forklar hvorfor.
Kildene kan være på et annet språk enn svarspråket; svar likevel på det språket som er angitt til slutt.

Svar KUN i dette JSON-formatet (ingen tekst utenfor JSON):
{{"items": [
  {{
    "label": "Konklusjon: <Tillatt | Ikke tillatt | Omfattes | Omfattes ikke | Betinget | Uklart> — kort kjerne i svaret",
    "description": "Begrunnelse i 1-3 setninger forankret i kildene. Ta med eventuelle vilkår eller unntak. Avslutt med 'Henvisning: <artikkel/avsnitt i Koden, resolusjon, eller Baby-Friendly-trinn>'.",
    "sources": ["Tittel på kilden(e) som forankrer svaret"]
  }}
]}}
Bruk normalt ETT item som direkte besvarer spørsmålet. Bruk flere kun når spørsmålet har klart adskilte deler.""",

        "aggregate_prompt": """Spørsmål: {question}
Relevante utdrag fra {n_docs} kilder:
{all_findings}
Gi ett tydelig compliance-svar på spørsmålet i JSON-formatet som beskrevet.""",

        "default_question": "Er dette tillatt etter WHO-koden, og hva er i så fall vilkårene?",
        "output_key": "findings",
    },
}


# ── Reading a structured template's shape ─────────────────────────────────────
# A "structured" query type asks each document for a JSON object rather than a
# bullet list, and asks the synthesis for one too. Which keys those objects hold
# is declared in the config — `doc_fields`, `doc_notes`, `agg_item_fields`,
# `agg_top_fields` — so the nodes below never name a key of their own. Strategisk
# risiko declares its chain that way, and a template built in the wizard declares
# whatever its author asked for; both run this same code.

def _spec_fields(cfg: Any, key: str) -> list[dict]:
    """The declared fields under `key`, skipping anything without a JSON key."""
    raw = (cfg or {}).get(key) or []
    return [f for f in raw if isinstance(f, dict) and (f.get("key") or "").strip()]


def _context_fields(cfg: Any) -> list[dict]:
    """Per-document fields that count as a contribution and are handed to the
    synthesis. `context: False` keeps a field in the report but out of both —
    open questions, for instance, are for the reader, not for merging."""
    return [f for f in _spec_fields(cfg, "doc_fields") if f.get("context", True)]


def _flat_finding_keys(cfg: Any) -> list[str]:
    """Which per-document fields stand in for the document in a flat list, in
    order of preference: the declared order, so the top field — what the
    document was read for — wins, and the next one with content fills in when
    it is empty. The same field the views show under a finding."""
    explicit = [k for k in ((cfg or {}).get("flat_from") or []) if isinstance(k, str)]
    if explicit:
        return explicit
    return [f["key"] for f in _spec_fields(cfg, "doc_fields")]


def _agg_items_key(cfg: Any) -> str:
    """Where the synthesis puts its list of findings."""
    return ((cfg or {}).get("agg_items_key") or "").strip() or (
        "temaer" if (cfg or {}).get("structured") else "items"
    )


# ── Output language ───────────────────────────────────────────────────────────
# Every prompt above is written in Norwegian. Rather than keep a translated copy
# of each query type, the chosen language is appended as a directive to whichever
# system prompt is in effect — that also covers user-edited prompts, which carry
# no placeholder of their own.

LANGUAGES = {
    "no": ("norsk bokmål", "norsk bokmål"),
    "nn": ("nynorsk",      "nynorsk"),
    "en": ("engelsk",      "English"),
    "sv": ("svensk",       "svenska"),
    "da": ("dansk",        "dansk"),
    "de": ("tysk",         "Deutsch"),
    "fr": ("fransk",       "français"),
    "es": ("spansk",       "español"),
    "it": ("italiensk",    "italiano"),
    "pl": ("polsk",        "polski"),
    "ar": ("arabisk",      "العربية"),
    "so": ("somali",       "Soomaali"),
    "uk": ("ukrainsk",     "українська"),
}

DEFAULT_LANGUAGE = "no"

# Appended verbatim, so it has to stay brace-free: callers add it *after*
# .format() has run on the template, where a stray {} would no longer be escaped.
# The INGEN RELEVANTE FUNN carve-out matters — extract_per_document tests for
# that literal string, so a translated version would be read as a real finding.
_LANGUAGE_DIRECTIVE = """

SPRÅK: Skriv alt du produserer på %s.
Dette gjelder både etiketter, overskrifter og brødtekst.
Oversett ikke JSON-nøkler eller feltnavn — bare verdiene.
Direkte sitater fra kildene kan beholdes på originalspråket, men din egen tekst skal være på %s.
UNNTAK: svarer du at dokumentet ikke er relevant, skriv nøyaktig INGEN RELEVANTE FUNN — den setningen skal aldri oversettes."""


def _language_directive(code: str) -> str:
    """Instruction appended to a system prompt so the model writes in `code`."""
    name, endonym = LANGUAGES.get(code or DEFAULT_LANGUAGE, LANGUAGES[DEFAULT_LANGUAGE])
    label = name if name == endonym else "%s (%s)" % (name, endonym)
    return _LANGUAGE_DIRECTIVE % (label, label)


# ── State ─────────────────────────────────────────────────────────────────────

class ChunkRef(BaseModel):
    page:    Optional[int] = None
    excerpt: str           = ""

class DocFindings(BaseModel):
    tittel:             str
    filename:           str
    kilde_url:          str            = ""
    kilde_type:         str            = ""   # "pdf" | "html" for materialized sources
    publisert_av:       str            = ""
    publisert_arstall:  Optional[int]  = None
    findings:           list[str]      = Field(default_factory=list)
    structured:         Optional[dict] = None   # set for "structured" query types
    chunks:             list[ChunkRef] = Field(default_factory=list)

class AggregateState(TypedDict):
    question: str
    query_type: str                     # "problems" | "moments" | "personas" | "free" | "strategisk_risiko"
    query_type_cfg: Any                 # effective (possibly user-edited) prompt config; falls back to QUERY_TYPES
    language: str                       # LANGUAGES key; output language for extraction + syntese
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
    extract_system = cfg["extract_system"] + _language_directive(
        state.get("language", DEFAULT_LANGUAGE))

    per_doc_findings: list[DocFindings] = []
    total_docs = len(state["documents"])

    _emit(state, {
        "event":      "node",
        "node":       "extract_per_document",
        "message":    f"Starter utvinning fra {total_docs} dokumenter…",
        "total_docs": total_docs,
    })

    cancel_event = state.get("cancel_event")
    # Per-document extraction is independent work, each dominated by one blocking
    # LLM call, so it runs concurrently rather than one document at a time.
    # Bounded to stay under Azure OpenAI rate limits (see EXTRACT_CONCURRENCY).
    max_workers = max(1, min(EXTRACT_CONCURRENCY, total_docs or 1))
    done_count = [0]                 # docs finished — drives the progress counter
    done_lock = threading.Lock()

    def _process_doc(doc_idx: int, entry: dict):
        """Extract findings from one document → (doc_idx, DocFindings | None).
        Runs in a worker thread; _emit is thread-safe via the SSE QueueProxy."""
        # A doc that starts after cancellation was requested does no work.
        if cancel_event is not None and cancel_event.is_set():
            return doc_idx, None

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
            return doc_idx, None

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
            return doc_idx, None

        print(f"  ↳ Context length: {len(context)} chars — calling LLM…", flush=True)

        prompt = cfg["extract_prompt"].format(
            question=question,
            tittel=tittel,
            context=context,
        )

        try:
            response = llm.invoke([
                SystemMessage(content=extract_system),
                HumanMessage(content=prompt),
            ])
            raw = (response.content or "").strip()
            print(f"  ↳ LLM response ({len(raw)} chars): {raw[:120]!r}", flush=True)
        except Exception as e:
            print(f"  ↳ LLM ERROR: {e}", flush=True)
            logging.warning("[aggregate] LLM failed for: %s", filename, exc_info=True)
            return doc_idx, None

        structured = None
        if cfg.get("structured"):
            try:
                structured = _parse_json_block(raw)
            except Exception as e:
                print(f"  ↳ JSON parse error: {e}", flush=True)
                logging.warning("[aggregate] structured parse failed for %s", filename, exc_info=True)
                return doc_idx, None
            if not structured or structured.get("relevant") is False:
                print(f"  ↳ Not relevant — skipping", flush=True)
                return doc_idx, None
            # Flat list for the summaries that show one line per document —
            # the template says which field speaks for the document.
            findings = []
            for flat_key in _flat_finding_keys(cfg):
                findings = [str(v) for v in (structured.get(flat_key) or []) if str(v).strip()]
                if findings:
                    break
            has_content = any(structured.get(f["key"]) for f in _context_fields(cfg))
        else:
            if "INGEN RELEVANTE FUNN" in raw.upper():
                print(f"  ↳ No relevant findings", flush=True)
                return doc_idx, None

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
        # Monotonic progress count — completions arrive out of order under
        # concurrency, so the running total, not doc_idx, is what advances.
        with done_lock:
            done_count[0] += 1
            completed = done_count[0]
        _emit(state, {
            "event":      "doc_done",
            "index":      doc_idx,
            "completed":  completed,
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
            logging.info("[aggregate] findings from: %s", tittel)
            return doc_idx, DocFindings(
                tittel=tittel,
                filename=filename,
                kilde_url=(entry.get("kilde_url") or ""),
                kilde_type=(entry.get("kilde_type") or ""),
                publisert_av=(entry.get("publisert_av") or ""),
                publisert_arstall=ar_int,
                findings=findings,
                structured=structured,
                chunks=chunks,
            )
        return doc_idx, None

    # Keep results keyed by index so the output stays in document order regardless
    # of the order tasks finish in.
    results: dict[int, DocFindings] = {}
    cancelled = False
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_doc, doc_idx, entry): doc_idx
            for doc_idx, entry in enumerate(state["documents"])
        }
        for fut in as_completed(futures):
            if cancel_event is not None and cancel_event.is_set() and not cancelled:
                cancelled = True
                # Not-yet-started docs are cancelled; in-flight ones run to the end.
                for pending in futures:
                    pending.cancel()
                logging.info("[aggregate] Cancellation requested — stopping after %d/%d docs",
                             done_count[0], total_docs)
                _emit(state, {
                    "event":   "cancelled",
                    "message": f"Avbrutt etter {done_count[0]}/{total_docs} dokumenter",
                    "index":   done_count[0],
                    "total":   total_docs,
                })
            try:
                doc_idx, df = fut.result()
            except CancelledError:
                continue
            if df is not None:
                results[doc_idx] = df

    per_doc_findings = [results[i] for i in sorted(results)]

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
    language = state.get("language", DEFAULT_LANGUAGE)
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

    all_findings_text = ""
    for doc in per_doc:
        all_findings_text += f"\n\n### {doc.tittel}\n"
        if structured and doc.structured:
            for field in _context_fields(cfg):
                vals = doc.structured.get(field["key"]) or []
                if vals:
                    label = field.get("label") or field["key"]
                    all_findings_text += f"{label}:\n" + "".join(f"- {v}\n" for v in vals)
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
            SystemMessage(content=cfg["aggregate_system"].format(n_personas=n_personas)
                                  + _language_directive(language)),
            HumanMessage(content=prompt),
        ])
        raw = (response.content or "").strip()
        print(f"[aggregate] Aggregation LLM response ({len(raw)} chars): {raw[:200]!r}", flush=True)
        parsed = _parse_json_block(raw)
        items = parsed.get(_agg_items_key(cfg), [])
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
        # Syntese-level fields that aren't per-item — the context around the list.
        for field in _spec_fields(cfg, "agg_top_fields"):
            result[field["key"]] = parsed.get(field["key"], [])
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