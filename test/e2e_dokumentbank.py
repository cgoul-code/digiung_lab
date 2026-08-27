#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end lifecycle check for a dokumentbank, run against a live server.

Walks one bank through the whole flow — create, add, build, add more, build,
delete, build, full rebuild, analyse, remove — and verifies each step through
the API rather than trusting the previous call's response. Everything happens in
a throwaway bank named `e2e_test_<timestamp>`, and the run ends by deleting it;
`--keep` leaves it behind when a failure needs inspecting.

The fixtures are PDFs generated in-process, so no sample files are needed. Each
carries a distinctive marker phrase, which is what lets the checks prove a
document really is (or no longer is) searchable instead of just counting rows.

    python test/e2e_dokumentbank.py                     # against Azure
    python test/e2e_dokumentbank.py --server http://localhost:80
    python test/e2e_dokumentbank.py --keep --verbose

Exit code is 0 only if every step passed.
"""
import argparse
import io
import sys
import time
import uuid

import requests

AZURE = "https://lab-document-query-g6djhxfnajdjgmbr.swedencentral-01.azurewebsites.net"

# Long: a build reads every document through an LLM and can idle behind a cold
# App Service instance.
BUILD_TIMEOUT = 900
ANALYSIS_TIMEOUT = 900
HTTP_TIMEOUT = 120


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_pdf(lines):
    """A one-page Helvetica PDF, written by hand so the agent needs no PDF
    library to produce its fixtures."""
    ops = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
    for i, line in enumerate(lines):
        esc = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        ops.append(f"({esc}) Tj" if i == 0 else f"T* ({esc}) Tj")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1", "replace")

    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return out.getvalue()


def fixture(marker, title, body_lines):
    """A document whose text contains `marker` — a nonsense word that cannot
    occur in any other document, so a hit on it is unambiguous."""
    return {
        "marker": marker,
        "filename": f"{marker.lower()}.pdf",
        "tittel": title,
        "bytes": make_pdf([title, f"Kjennemerke: {marker}."] + body_lines),
    }


def build_fixtures(run_id):
    """Four documents: one for the first build, three for the second."""
    def m(n):
        return f"KVASIMORF{n}{run_id}"
    return [
        fixture(m(1), "Rapport om skolefravaer blant ungdom", [
            "Utgitt av Testdirektoratet i 2021.",
            "Funn: mange unge opplever press knyttet til skole og sosiale medier.",
            "Tiltak: tidlig innsats og tverrfaglig samarbeid anbefales.",
        ]),
        fixture(m(2), "Kartlegging av psykisk helse i videregaaende skole", [
            "Utgitt av Testdirektoratet i 2022.",
            "Funn: ensomhet og soevnproblemer gaar igjen i intervjuene.",
            "Behov: lavterskeltilbud naer skolen.",
        ]),
        fixture(m(3), "Evaluering av digitale hjelpetjenester for unge", [
            "Utgitt av Testdirektoratet i 2023.",
            "Funn: unge foretrekker anonyme kanaler foer de tar kontakt.",
            "Utfordring: tjenestene er fragmenterte og lite kjente.",
        ]),
        fixture(m(4), "Ungdom og fysisk aktivitet i fritiden", [
            "Utgitt av Testdirektoratet i 2024.",
            "Funn: frafall fra organisert idrett skjer tidlig i tenaarene.",
            "Tiltak: billigere og mer fleksible lavterskeltilbud.",
        ]),
    ]


# ── Reporting ─────────────────────────────────────────────────────────────────

class Report:
    def __init__(self, verbose):
        self.verbose = verbose
        self.rows = []

    def log(self, msg):
        print(f"    {msg}", flush=True)

    def debug(self, msg):
        if self.verbose:
            print(f"      · {msg}", flush=True)

    def step(self, num, title):
        print(f"\n[{num}] {title}", flush=True)

    def record(self, name, ok, detail="", skipped=False):
        mark = "SKIP" if skipped else ("PASS" if ok else "FAIL")
        self.rows.append((mark, name, detail))
        print(f"    {mark}  {name}{(' — ' + detail) if detail else ''}", flush=True)
        return ok

    def summary(self):
        print("\n" + "=" * 72)
        failed = [r for r in self.rows if r[0] == "FAIL"]
        skipped = [r for r in self.rows if r[0] == "SKIP"]
        passed = [r for r in self.rows if r[0] == "PASS"]
        for mark, name, detail in self.rows:
            print(f"{mark:5} {name}{(' — ' + detail) if detail else ''}")
        print("=" * 72)
        print(f"{len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
        return len(failed) == 0


# ── API wrapper ───────────────────────────────────────────────────────────────

class Api:
    def __init__(self, base, report):
        self.base = base.rstrip("/")
        self.r = report

    def _call(self, method, path, **kw):
        kw.setdefault("timeout", HTTP_TIMEOUT)
        url = f"{self.base}{path}"
        self.r.debug(f"{method} {path}")
        return requests.request(method, url, **kw)

    def get(self, path, **kw):
        return self._call("GET", path, **kw)

    def post(self, path, **kw):
        return self._call("POST", path, **kw)

    def delete(self, path, **kw):
        return self._call("DELETE", path, **kw)

    def json(self, res):
        try:
            return res.json()
        except Exception:
            return {"_raw": res.text[:400]}

    # -- domain helpers --

    def indexes(self):
        res = self.get("/indexes")
        return self.json(res) if res.ok else []

    def entries(self, bank):
        res = self.get(f"/admin/entries?index_name={bank}")
        return self.json(res).get("entries", []) if res.ok else None

    def upload(self, bank, doc):
        files = {"file": (doc["filename"], doc["bytes"], "application/pdf")}
        data = {"tittel": doc["tittel"], "publisert_av": "Testdirektoratet",
                "segment": "e2e", "type_kilde": "rapport"}
        return self.post(f"/admin/entries?index_name={bank}", files=files, data=data)

    def reindex(self, bank, mode, timeout=BUILD_TIMEOUT):
        """Start a build and follow it to a terminal event. Returns
        (ok, status, events)."""
        res = self.post(f"/admin/reindex?index_name={bank}&mode={mode}")
        if not res.ok:
            return False, f"start failed: {res.status_code} {res.text[:200]}", []
        job_id = self.json(res).get("job_id")
        if not job_id:
            return False, "no job_id in response", []

        events, last, deadline = [], 0, time.time() + timeout
        while time.time() < deadline:
            time.sleep(2)
            poll = self.get(f"/admin/reindex/{job_id}?last={last}")
            if poll.status_code == 404:
                return False, "job disappeared from the server", events
            if not poll.ok:
                continue
            data = self.json(poll)
            new = data.get("events", [])
            for ev in new:
                self.r.debug(f"{ev.get('event')}: {ev.get('tittel') or ev.get('message') or ''}")
            events += new
            last = data.get("total", last)
            status = data.get("status")
            if any(e.get("event") == "done" for e in events):
                return True, "done", events
            if any(e.get("event") == "error" for e in events) or status == "error":
                msg = next((e.get("message") for e in events if e.get("event") == "error"), "error")
                return False, msg, events
            if status in ("done",):
                return True, "done", events
        return False, f"timed out after {timeout}s", events

    def query(self, bank, question, top_k=20):
        """Returns (data, err). A build reloads the index, and the server can
        briefly report not-ready straight afterwards, so one retry."""
        for attempt in (1, 2):
            res = self.post("/query", json={"question": question, "index_name": bank,
                                            "top_k": top_k, "cutoff": 0.0})
            if res.ok:
                return self.json(res), None
            if attempt == 1 and res.status_code in (503, 502, 504):
                time.sleep(5)
                continue
            return None, f"HTTP {res.status_code} {res.text[:160]}"
        return None, "unreachable"

    def pending(self, bank):
        """What a build still has to do, from the ingest manifest."""
        res = self.get(f"/admin/reindex/pending?index_name={bank}")
        if not res.ok:
            return None, f"HTTP {res.status_code}"
        d = self.json(res)
        return (len(d.get("to_ingest") or []), d.get("to_prune", 0)), None

    def analyse(self, bank, question, query_type="free", timeout=ANALYSIS_TIMEOUT):
        res = self.post("/aggregate/stream", json={
            "question": question, "query_type": query_type,
            "index_name": bank, "chunks_per_doc": 2, "include_aggregate": True,
        })
        if not res.ok:
            return None, f"start failed: {res.status_code} {res.text[:200]}"
        job_id = self.json(res).get("job_id")
        if not job_id:
            return None, "no job_id in response"

        last, deadline = 0, time.time() + timeout
        while time.time() < deadline:
            time.sleep(2)
            poll = self.get(f"/aggregate/stream/{job_id}?last={last}")
            if not poll.ok:
                continue
            data = self.json(poll)
            for ev in data.get("events", []):
                kind = ev.get("event")
                self.r.debug(f"{kind}: {ev.get('message') or ev.get('tittel') or ''}")
                if kind == "result":
                    return ev, None
                if kind == "error":
                    return None, ev.get("message", "error")
                if kind == "cancelled":
                    return None, "cancelled"
            last = data.get("total", last)
            if data.get("status") == "error":
                return None, "job reported error"
        return None, f"timed out after {timeout}s"


def cited_files(result):
    """Filenames the answer actually drew on. Only `sources` counts: the
    response also echoes the question, and the model tends to repeat wording
    from it in the answer, so anything wider matches itself."""
    return {(s.get("filename") or "").lower() for s in (result or {}).get("sources") or []}


# ── The run ───────────────────────────────────────────────────────────────────

def run(args):
    rep = Report(args.verbose)
    api = Api(args.server, rep)
    run_id = uuid.uuid4().hex[:6].upper()
    bank = args.bank or f"e2e_test_{int(time.time())}"
    docs = build_fixtures(run_id)
    created = False

    print(f"Server : {api.base}")
    print(f"Bank   : {bank}")
    print(f"Run id : {run_id}")

    # Reachability and capabilities, before anything is created.
    rep.step(0, "Serveren svarer")
    health = api.get("/health")
    if not rep.record("GET /health", health.ok, f"HTTP {health.status_code}"):
        return rep.summary()
    hdata = api.json(health)
    rep.log(f"state={hdata.get('state')} indekser={len(hdata.get('loaded') or [])}")

    # An empty name is rejected with 400 by the endpoint itself; a server
    # without it answers 405 (the path exists for POST only) or 404.
    probe = api.delete("/admin/indexes?index_name=&confirm=")
    can_delete_bank = probe.status_code == 400
    if not can_delete_bank:
        rep.log("MERK: serveren har ikke DELETE /admin/indexes — steg 8 kan ikke kjøres.")

    try:
        # 1 ── create
        rep.step(1, "Opprett ny dokumentbank")
        res = api.post("/admin/indexes", json={"name": bank, "query_types": ["free"]})
        created = res.ok
        rep.record("POST /admin/indexes", res.ok, f"HTTP {res.status_code} {api.json(res)}")
        rep.record("banken er i /indexes", bank in api.indexes())
        rep.record("banken er tom", api.entries(bank) == [])

        # 2 ── add one document
        rep.step(2, "Legg til ett dokument fra lokalt filområde")
        up = api.upload(bank, docs[0])
        rep.record("POST /admin/entries", up.ok, f"HTTP {up.status_code}")
        rep.record("listen har 1 oppføring", len(api.entries(bank) or []) == 1)

        # 3 ── build
        rep.step(3, "Oppdater dokumentbanken")
        ok, status, _ = api.reindex(bank, "incremental")
        rep.record("inkrementell bygging", ok, status)
        counts, perr = api.pending(bank)
        rep.record("manifestet er ajour", counts == (0, 0), perr or f"to_ingest/to_prune={counts}")
        hit, qerr = api.query(bank, "Hva sier dokumentet om skolefravaer blant ungdom?")
        rep.record("dokumentet er soekbart", docs[0]["filename"] in cited_files(hit),
                   qerr or f"kilder={sorted(cited_files(hit))}")

        # 4 ── add three more, build again
        rep.step(4, "Legg til 3 nye dokumenter og oppdater")
        for d in docs[1:]:
            r = api.upload(bank, d)
            rep.record(f"lastet opp {d['filename']}", r.ok, f"HTTP {r.status_code}")
        rep.record("listen har 4 oppføringer", len(api.entries(bank) or []) == 4)

        ok, status, events = api.reindex(bank, "incremental")
        rep.record("inkrementell bygging", ok, status)
        skipped = sum(1 for e in events if e.get("event") == "skip")
        rep.record("den første ble hoppet over", skipped >= 1,
                   f"{skipped} skip-hendelser — bare de nye leses")
        counts, perr = api.pending(bank)
        rep.record("manifestet er ajour", counts == (0, 0), perr or f"to_ingest/to_prune={counts}")
        hit, qerr = api.query(bank, "Hva sier dokumentene om frafall fra organisert idrett?")
        rep.record("nytt dokument er soekbart", docs[3]["filename"] in cited_files(hit),
                   qerr or f"kilder={sorted(cited_files(hit))}")

        # 5 ── delete one, build again
        rep.step(5, "Slett ett dokument og oppdater")
        victim = docs[1]
        before = api.entries(bank) or []
        key = next((e.get("url") or e.get("filnavn") for e in before
                    if (e.get("filnavn") or "").endswith(victim["filename"])), None)
        rep.record("fant oppføringen som skal slettes", key is not None, str(key))
        if key:
            d = api.delete(f"/admin/entries?index_name={bank}&key={requests.utils.quote(key)}")
            rep.record("DELETE /admin/entries", d.ok, f"HTTP {d.status_code}")
        rep.record("listen har 3 oppføringer", len(api.entries(bank) or []) == 3)

        counts, perr = api.pending(bank)
        rep.record("banken vet at ett dokument skal fjernes", counts == (0, 1),
                   perr or f"to_ingest/to_prune={counts}")

        ok, status, events = api.reindex(bank, "incremental")
        rep.record("inkrementell bygging", ok, status)
        pruned = [e for e in events if e.get("event") == "pruned"]
        rep.record("byggingen ryddet bort dokumentet", bool(pruned),
                   pruned[0].get("message") if pruned else "ingen pruned-hendelse")
        counts, perr = api.pending(bank)
        rep.record("ingenting gjenstaar etter oppryddingen", counts == (0, 0),
                   perr or f"to_ingest/to_prune={counts}")
        hit, qerr = api.query(bank, "Hva sier dokumentene om ensomhet og soevnproblemer?")
        rep.record("slettet dokument gir ingen treff", victim["filename"] not in cited_files(hit),
                   qerr or f"kilder={sorted(cited_files(hit))}")

        # 6 ── full rebuild
        rep.step(6, "Regenerer hele dokumentbanken")
        ok, status, _ = api.reindex(bank, "full")
        rep.record("full ombygging", ok, status)
        rep.record("listen er uendret (3)", len(api.entries(bank) or []) == 3)
        counts, perr = api.pending(bank)
        rep.record("manifestet er ajour", counts == (0, 0), perr or f"to_ingest/to_prune={counts}")
        hit, qerr = api.query(bank, "Hva sier dokumentet om skolefravaer blant ungdom?")
        rep.record("gjenvaerende dokument er soekbart", docs[0]["filename"] in cited_files(hit),
                   qerr or f"kilder={sorted(cited_files(hit))}")
        hit, qerr = api.query(bank, "Hva sier dokumentene om ensomhet og soevnproblemer?")
        rep.record("slettet dokument er fortsatt borte", victim["filename"] not in cited_files(hit),
                   qerr or f"kilder={sorted(cited_files(hit))}")


        # 7 ── analysis
        rep.step(7, "Kjør en analyse")
        result, err = api.analyse(bank, "Hva er gjennomgaaende i materialet?")
        rep.record("aggregert analyse fullfoerte", result is not None, err or "")
        if result:
            visited = result.get("documents_visited")
            rep.record("analysen besoekte 3 dokumenter", visited == 3, f"documents_visited={visited}")
            findings = result.get("findings") or result.get("risikoomrader") or []
            rep.record("analysen ga resultater", len(findings) > 0, f"{len(findings)} funn")

    finally:
        # 8 ── remove the bank
        rep.step(8, "Fjern dokumentbanken")
        if args.keep:
            rep.record("opprydding", True, f"--keep: «{bank}» er beholdt", skipped=True)
        elif not created:
            rep.record("opprydding", True, "banken ble aldri opprettet", skipped=True)
        elif not can_delete_bank:
            # Can't remove the bank itself, but leaving four test documents in a
            # live list is worse than leaving an empty bank behind.
            emptied = 0
            for e in (api.entries(bank) or []):
                k = e.get("url") or e.get("filnavn")
                if k and api.delete(
                        f"/admin/entries?index_name={bank}&key={requests.utils.quote(k)}").ok:
                    emptied += 1
            rep.record("DELETE /admin/indexes", False,
                       f"endepunktet finnes ikke på serveren — «{bank}» må fjernes manuelt "
                       f"(tømt for {emptied} dokumenter)")
        else:
            d = api.delete(f"/admin/indexes?index_name={bank}&confirm={bank}")
            rep.record("DELETE /admin/indexes", d.ok, f"HTTP {d.status_code} {api.json(d)}")
            rep.record("banken er borte fra /indexes", bank not in api.indexes())

    return rep.summary()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default=AZURE, help="server base URL (default: Azure)")
    p.add_argument("--bank", default=None, help="bank name (default: e2e_test_<timestamp>)")
    p.add_argument("--keep", action="store_true", help="leave the bank behind for inspection")
    p.add_argument("--verbose", "-v", action="store_true", help="log every request and event")
    args = p.parse_args()

    started = time.time()
    ok = run(args)
    print(f"Brukte {time.time() - started:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
