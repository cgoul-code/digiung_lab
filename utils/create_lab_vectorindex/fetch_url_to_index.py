"""One-shot: download a URL from THIS machine and upload it to a running server.

Why this exists: regjeringen.no (and similar WAF-fronted sites) return HTTP 403
to the Azure datacenter IP, so the server can't fetch them itself. A browser SPA
can't do it either — the site sends no CORS header, so cross-origin fetch+read is
blocked. But your own machine gets HTTP 200, so we download here and POST the
bytes to the server's /admin/entries endpoint as a file upload (which stores it on
the index file share — no server-side fetch involved).

One operation:
    python -m utils.create_lab_vectorindex.fetch_url_to_index \
        --server https://<app-host> --index Testindeks \
        --url https://www.regjeringen.no/.../stm....pdf --tittel "Perspektivmeldingen"

To swap an existing (un-fetchable) URL entry for the downloaded file, add
    --replace-url https://www.regjeringen.no/.../stm....pdf
The server carries over that entry's metadata and keeps the URL as kilde_url.
"""

import argparse
import os
import sys
import time
from urllib.parse import urlsplit, unquote

import requests

# Browser-like headers — same intent as ingest_pdfs._BROWSER_HEADERS (kept local
# to avoid importing the heavy llama-index stack just to download a file).
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "nb-NO,nb;q=0.9,no;q=0.8,en;q=0.7",
}

# Optional metadata flags → document_store fields.
_META_FLAGS = ["tittel", "publisert_av", "publisert_arstall", "segment",
               "type_kilde", "malgruppe", "antall_deltakere", "oppsummering"]


def _derive_filename(url: str, content_type: str) -> str:
    name = os.path.basename(unquote(urlsplit(url).path)) or ""
    if not name.lower().endswith((".pdf", ".pptx", ".ppt")):
        if "pdf" in content_type:
            name = (name or "document") + ".pdf"
    return name or "document.pdf"


def main() -> int:
    p = argparse.ArgumentParser(description="Download a URL locally and upload it to the server in one step.")
    p.add_argument("--server", required=True, help="Base URL of the running app, e.g. https://myapp.azurewebsites.net")
    p.add_argument("--index", required=True, help="Target index name (e.g. Testindeks)")
    p.add_argument("--url", required=True, help="URL of the PDF/PPTX to download")
    p.add_argument("--replace-url", default="", help="Existing URL entry key to replace with the uploaded file")
    p.add_argument("--filename", default="", help="Override the stored filename")
    p.add_argument("--reindex", nargs="?", const="incremental", default=None,
                   choices=["incremental", "full"],
                   help="After upload, trigger (and follow) a reindex. Default mode: incremental")
    for f in _META_FLAGS:
        p.add_argument(f"--{f.replace('_', '-')}", dest=f, default=None)
    args = p.parse_args()

    # 1. Download from THIS machine (the IP the site allows).
    print(f"↓ Downloading {args.url}", flush=True)
    try:
        r = requests.get(args.url, headers=_BROWSER_HEADERS, timeout=120)
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"  ✗ Download failed: {e}", file=sys.stderr)
        return 1
    content_type = (r.headers.get("Content-Type") or "").lower()
    fname = args.filename or _derive_filename(args.url, content_type)
    if not fname.lower().endswith((".pdf", ".pptx", ".ppt")):
        print(f"  ✗ Not a PDF/PPTX (Content-Type: {content_type or 'unknown'}). "
              f"This tool only handles document files, not HTML pages.", file=sys.stderr)
        return 1
    print(f"  ✓ {len(r.content):,} bytes ({content_type or 'unknown'}) → {fname}", flush=True)

    # 2. Upload to the server as a file entry (stored on the index file share).
    data = {"kilde_url": args.url}
    for f in _META_FLAGS:
        v = getattr(args, f)
        if v not in (None, ""):
            data[f] = v
    if args.replace_url:
        data["replace_key"] = args.replace_url

    endpoint = f"{args.server.rstrip('/')}/admin/entries?index_name={requests.utils.quote(args.index)}"
    print(f"↑ Uploading to {endpoint}", flush=True)
    resp = requests.post(endpoint, files={"file": (fname, r.content, content_type or "application/pdf")},
                         data=data, timeout=180)
    if not resp.ok:
        detail = ""
        try:
            detail = resp.json().get("error", "")
        except Exception:
            detail = resp.text[:300]
        print(f"  ✗ Upload failed: HTTP {resp.status_code} {detail}", file=sys.stderr)
        return 1

    entry = resp.json().get("entry", {})
    print(f"  ✓ Stored as '{entry.get('filnavn', fname)}' in index '{args.index}'.", flush=True)

    if not args.reindex:
        print("Now run a reindex for this index (admin → Regenerer) — no server-side fetch needed.", flush=True)
        return 0

    # 3. Optionally trigger + follow the reindex so it's one end-to-end operation.
    base = args.server.rstrip("/")
    idx_q = requests.utils.quote(args.index)
    print(f"↻ Triggering reindex (mode={args.reindex})", flush=True)
    rr = requests.post(f"{base}/admin/reindex?index_name={idx_q}&mode={args.reindex}", timeout=60)
    if not rr.ok:
        print(f"  ✗ Reindex trigger failed: HTTP {rr.status_code} {rr.text[:200]}", file=sys.stderr)
        return 1
    job_id = rr.json().get("job_id")
    seen = 0
    while True:
        time.sleep(2)
        jr = requests.get(f"{base}/admin/reindex/{job_id}", timeout=60)
        if not jr.ok:
            print(f"  ✗ Reindex poll failed: HTTP {jr.status_code}", file=sys.stderr)
            return 1
        job = jr.json()
        events = job.get("events", [])
        for ev in events[seen:]:
            msg = ev.get("message") or ev.get("tittel") or ""
            print(f"    [{ev.get('event')}] {msg}".rstrip(), flush=True)
        seen = len(events)
        if job.get("status") in ("done", "error"):
            print(f"  Reindex {job.get('status')}.", flush=True)
            return 0 if job.get("status") == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
