"""
materialize_sources.py
======================
Turn every URL entry in document_store.json into a local PDF file in the data
area, then rewrite the entry to a file entry. After this, all sources are PDFs
on disk (and in blob), so ingestion never needs to fetch a URL again.

  - PDF URLs        → downloaded as-is
  - HTML pages      → main text extracted and rendered to PDF (xhtml2pdf)
  - SPA topic pages → topic text extracted and rendered to PDF
                      (helsenorge .../hvaerinnafor/#/temabeskrivelse/<slug>)

Each rewritten entry becomes:
    {
      "filnavn":    "data/<index>/<slug>.pdf",   # local, matches existing convention
      "kilde_url":  "<original url>",             # kept so citations still link out
      "kilde_type": "pdf" | "html",              # how to deep-link the citation
      ...all other metadata preserved (tittel, publisert_av, …); url/method dropped
    }

WHY RUN THIS LOCALLY
--------------------
Downloading is an outbound HTTP request from whichever machine runs this code.
Some sites (e.g. regjeringen.no behind a CDN) return HTTP 403 to Azure's
datacenter IP but 200 to a normal client. So run this where the URLs are
reachable (your machine). With blob sync enabled it uploads the PDFs + the
updated document_store.json to the container, so the Azure app picks them up.

LOCAL-ONLY DEPENDENCY (not needed on the Azure server, which only reads PDFs):
    pip install "xhtml2pdf" "cryptography<49"

USAGE
-----
    # Dry run — show what would happen, change nothing:
    python -m utils.create_lab_vectorindex.materialize_sources --index Strategisk_risiko --dry-run

    # Do it (writes PDFs + rewrites document_store.json):
    python -m utils.create_lab_vectorindex.materialize_sources --index Strategisk_risiko

    # Also push PDFs + document_store.json to blob (needs CONNECTION_STRING):
    USE_BLOB_INDEXES=true python -m utils.create_lab_vectorindex.materialize_sources --index Strategisk_risiko
"""

import argparse
import hashlib
import html as _html
import io
import json
import logging
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit, unquote

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import requests
from bs4 import BeautifulSoup

# Reuse the fetch helpers (browser headers, SPA extraction) from the ingest module.
from utils.create_lab_vectorindex.ingest_pdfs import (
    _BROWSER_HEADERS,
    _spa_topic_text,
    _SPA_PAGE_CACHE,  # noqa: F401  (kept warm across SPA fetches)
)

# Optional blob sync — only active when azure_blob.ENABLED (USE_BLOB_INDEXES / IN_AZURE).
try:
    import azure_blob
except Exception:  # pragma: no cover - azure_blob lives at the project root
    azure_blob = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_SPA_MARKER = "/temabeskrivelse/"


# ── filename + PDF helpers ─────────────────────────────────────────────────────

def _slug_for(url: str) -> str:
    """A safe, stable, unique .pdf filename derived from the URL."""
    parts = urlsplit(url)
    # SPA topic URLs carry the distinguishing slug in the fragment, not the path.
    if parts.fragment:
        base = unquote(parts.fragment).rstrip("/").split("/")[-1]
    else:
        base = unquote(os.path.basename(parts.path.rstrip("/")))
    base = base or "side"
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "side"
    base = base[:80]
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{digest}.pdf"


def _html_to_pdf_bytes(title: str, blocks: list[str]) -> bytes:
    """Render a title + text blocks to a simple, clean PDF via xhtml2pdf."""
    from xhtml2pdf import pisa  # local-only dependency

    parts = [
        "<html><head><meta charset='utf-8'><style>"
        "@page{size:a4;margin:2cm} body{font-family:Helvetica;font-size:11pt;line-height:1.45}"
        "h1{font-size:16pt;margin:0 0 12pt 0} p{margin:0 0 8pt 0}"
        "</style></head><body>"
    ]
    if title:
        parts.append(f"<h1>{_html.escape(title)}</h1>")
    for blk in blocks:
        blk = blk.strip()
        if blk:
            parts.append(f"<p>{_html.escape(blk)}</p>")
    parts.append("</body></html>")

    buf = io.BytesIO()
    result = pisa.CreatePDF(src="".join(parts), dest=buf, encoding="utf-8")
    if result.err:
        raise RuntimeError(f"xhtml2pdf reported {result.err} error(s)")
    data = buf.getvalue()
    if not data.startswith(b"%PDF-"):
        raise RuntimeError("xhtml2pdf produced no valid PDF")
    return data


# ── fetch one URL → (pdf_bytes, kilde_type, page_title) ────────────────────────

def _fetch_as_pdf(url: str) -> tuple[bytes, str, str]:
    """Return (pdf_bytes, kilde_type, page_title) for a URL.

    kilde_type is "pdf" when the URL itself is a PDF, else "html".
    Raises on failure so the caller can skip the entry.
    """
    fragment = urlsplit(url).fragment
    if fragment and _SPA_MARKER in fragment:
        title, body = _spa_topic_text(url)
        blocks = [b for b in body.split("\n\n") if b.strip()]
        return _html_to_pdf_bytes(title, blocks), "html", title

    resp = requests.get(url, timeout=60, headers=_BROWSER_HEADERS)
    resp.raise_for_status()

    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "application/pdf" in content_type or resp.content[:5] == b"%PDF-":
        return resp.content, "pdf", ""

    # HTML page → extract main readable text, render to PDF.
    if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding or "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    blocks = [ln.strip() for ln in main.get_text(separator="\n", strip=True).splitlines() if ln.strip()]
    if not blocks:
        raise ValueError(f"No textual content extracted from {url}")

    page_title = ""
    if soup.title and soup.title.string:
        page_title = soup.title.string.strip()
    elif soup.find("h1"):
        page_title = soup.find("h1").get_text(strip=True)
    return _html_to_pdf_bytes(page_title, blocks), "html", page_title


# ── main ───────────────────────────────────────────────────────────────────────

def materialize_index(document_store_path: str, index_name: str, data_dir: str,
                      dry_run: bool = False, push_blob: bool = True) -> dict:
    """Download/convert every URL entry for `index_name` into a PDF under
    <data_dir>/<index_name>/ and rewrite the entry to a file entry.

    Returns a summary dict. Writes nothing when dry_run is True.
    """
    blob_on = bool(push_blob and azure_blob is not None and getattr(azure_blob, "ENABLED", False))

    # Work on the canonical store: pull the latest from blob first, so an index
    # created in Azure (not present locally) is still found.
    if blob_on and not dry_run:
        if azure_blob.download_file("document_store.json", document_store_path):
            logging.info("Pulled latest document_store.json from blob.")

    with open(document_store_path, "r", encoding="utf-8") as f:
        store = json.load(f)
    if not isinstance(store, dict) or index_name not in store:
        raise SystemExit(f"Index '{index_name}' not found in {document_store_path}")

    entries = store[index_name]
    dest_dir = os.path.join(data_dir, index_name)
    if not dry_run:
        os.makedirs(dest_dir, exist_ok=True)

    summary = {"converted": 0, "skipped_file": 0, "failed": 0, "failures": []}

    for i, entry in enumerate(entries):
        url = entry.get("url")
        if not url:
            summary["skipped_file"] += 1
            continue  # already a file entry

        tittel = entry.get("tittel") or url
        filename = _slug_for(url)
        rel_path = f"data/{index_name}/{filename}"
        abs_path = os.path.join(dest_dir, filename)

        logging.info("[%d/%d] %s", i + 1, len(entries), tittel)
        logging.info("        %s", url)
        try:
            pdf_bytes, kilde_type, page_title = _fetch_as_pdf(url)
        except Exception as e:
            logging.warning("        ✗ FAILED: %s", e)
            summary["failed"] += 1
            summary["failures"].append({"url": url, "error": str(e)})
            continue

        logging.info("        → %s  (%s, %d KB)", rel_path, kilde_type, len(pdf_bytes) // 1024)
        if dry_run:
            summary["converted"] += 1
            continue

        with open(abs_path, "wb") as fp:
            fp.write(pdf_bytes)
        if blob_on:
            azure_blob.upload_data_file(abs_path, index_name, filename)

        # Rewrite entry → file entry, preserving metadata; keep original URL as kilde_url.
        new_entry = {k: v for k, v in entry.items() if k not in ("url", "method")}
        new_entry["filnavn"] = rel_path
        new_entry["kilde_url"] = entry.get("kilde_url") or url
        new_entry["kilde_type"] = kilde_type
        if not new_entry.get("tittel") and page_title:
            new_entry["tittel"] = page_title
        entries[i] = new_entry
        summary["converted"] += 1

    if not dry_run and summary["converted"]:
        store[index_name] = entries
        tmp = document_store_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
        os.replace(tmp, document_store_path)
        logging.info("Updated %s", document_store_path)
        if blob_on:
            azure_blob.upload_document_store(document_store_path)
            logging.info("Uploaded document_store.json + data files to blob.")

    logging.info(
        "Done: %d converted, %d already files, %d failed%s",
        summary["converted"], summary["skipped_file"], summary["failed"],
        " (DRY RUN — nothing written)" if dry_run else "",
    )
    if summary["failures"]:
        logging.warning("Failures:")
        for fl in summary["failures"]:
            logging.warning("  ✗ %s — %s", fl["url"], fl["error"])
    return summary


def _default_data_dir() -> str:
    """Mirror query_server's local data-dir resolution."""
    in_azure = "WEBSITE_SITE_NAME" in os.environ or "FUNCTIONS_WORKER_RUNTIME" in os.environ
    return ("" if in_azure else ".") + os.getenv("DATA_DIR", "/data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Materialize URL sources into local PDF files")
    parser.add_argument("--index", required=True, help="Index name (key in document_store.json)")
    parser.add_argument("--document_store",
                        default="./utils/create_lab_vectorindex/document_store.json",
                        help="Path to document_store.json")
    parser.add_argument("--data-dir", default=_default_data_dir(), help="Data area root")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen; change nothing")
    parser.add_argument("--no-blob", action="store_true", help="Do not upload to blob even if enabled")
    args = parser.parse_args()

    materialize_index(
        document_store_path=args.document_store,
        index_name=args.index,
        data_dir=args.data_dir,
        dry_run=args.dry_run,
        push_blob=not args.no_blob,
    )
