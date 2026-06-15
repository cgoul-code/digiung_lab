"""Cluster index TextNodes by URL topic, then compute discriminative keywords per cluster.

For each topical cluster we pick the unigrams/bigrams that appear MUCH more often
inside the cluster than in the rest of the corpus (log-odds with smoothing).
"""
import json, re
from pathlib import Path
from collections import Counter, defaultdict
from urllib.parse import unquote
from math import log

INDEX_DIR = Path("./blobstorage/chatbot/hvaerinnafor")

with open(INDEX_DIR / "docstore.json", "r", encoding="utf-8") as f:
    docstore = json.load(f)

# (source, text) per chunk
chunks = []
for nid, payload in docstore.get("docstore/data", {}).items():
    inner = payload.get("__data__") or payload
    text  = (inner.get("text") or "").strip()
    meta  = inner.get("metadata") or {}
    src   = unquote(meta.get("url") or meta.get("source_file") or meta.get("filename") or "")
    if text:
        chunks.append((src, text))

# Classify each source URL into one cluster.
# Pattern → cluster name (Norwegian, matches user's style)
CLUSTERS = [
    # (label, list of regex matched against the source URL)
    ("Forelskelse, følelser & usikkerhet", [
        r"/forelskelse/", r"vanlig.?folelse", r"vanlig.?følelse",
    ]),
    ("Flørting, dating & å bli kjent", [
        r"florte", r"flørte", r"snapchat", r"daten", r"liker.deg",
        r"a_like_deg", r"å_like_deg",
    ]),
    ("Parforhold, kommunikasjon & brudd", [
        r"/forhold/", r"eksen", r"brudd", r"kjeresten", r"kjæresten",
    ]),
    ("Pubertet & kroppsutvikling", [
        r"/kropp/", r"/pubertet", r"pubertet",
    ]),
    ("Sex, lyst & onanering", [
        r"/Sex/", r"onan", r"kat", r"kåt", r"sexlyst",
    ]),
    ("Samtykke, grenser & trygghet", [
        r"samtykke", r"grenser", r"hva_sier_loven", r"aldersforskjell.lov",
        r"hvor_stor_aldersforskjell",
    ]),
    ("Nakenbilder, deling & nettrelaterte risikoer", [
        r"nakenbilder", r"bilder", r"sende_", r"video",
    ]),
    ("Porno & seksuelle fantasier", [
        r"porno", r"hentai", r"fantas",
    ]),
    ("Overgrep, vold & forebygging", [
        r"/overgrep/", r"/vold/", r"har-du-utsatt", r"pedofil",
        r"hva_om_jeg_fantaserer", r"blotting", r"sex_med_dyr",
        r"sex_med_barn", r"sex_med_noen_i_familien",
    ]),
    ("Skeiv identitet & mangfold", [
        r"/Homofil/", r"jeg-er-skeiv", r"skeiv", r"Funksjonsnedsettelser",
        r"pride",
    ]),
    ("Seksuelt overførbare infeksjoner", [
        r"kjonnssykdom", r"klamydia", r"gonore", r"herpes", r"hpv",
        r"kjonnsvorter", r"skabb",
    ]),
]

def cluster_for(url: str) -> str:
    for label, patterns in CLUSTERS:
        for pat in patterns:
            if re.search(pat, url, re.IGNORECASE):
                return label
    return "_Other_"

# Group text per cluster
by_cluster: dict[str, list[str]] = defaultdict(list)
sources_per_cluster: dict[str, list[str]] = defaultdict(list)
for src, text in chunks:
    c = cluster_for(src)
    by_cluster[c].append(text)
    if src not in sources_per_cluster[c]:
        sources_per_cluster[c].append(src)

print(f"chunks={len(chunks)}  clusters={len(by_cluster)}")
for label, texts in by_cluster.items():
    print(f"  {label:50s}  chunks={len(texts):3d}  sources={len(sources_per_cluster[label]):3d}")
print()

# Tokenize
STOP = set("""
og er i det som på til av de den at han hun jeg du vi dere hvis for med et en ei har kan skal vil ikke
være var blir ble blitt hva når hvor hvordan hvorfor alle noen mange mer mest også bare om så slik selv
seg sin sitt sine din ditt dine deres fra ut opp inn over under etter før mellom ved mot da men eller
fordi jo nei ja kanskje heller mindre veldig alltid aldri ofte sjelden gjerne klart helt ganske litt mye
nok både like samme andre ene ny nye gammel gamle godt god gode bra dårlig store stor lille liten her
der dette disse denne blant gjennom rundt uten mens sånn sånt sånne deg meg oss dem henne ham vet fått
få får gjøre gjør gjorde kommer kom gå går gikk snakker snakke sier si sa tenker tenke tenkte føler føle
følte lurer lure lurte mente mener tror tro trodde ser se så samtidig derfor altså mens mest minste større
mindre tema temaer fag faget faglig sist oppdatert skrevet redaksjonen samarbeid sexologisk rådgiver
ungdomsjournalist colourbox foto første andre tredje nye gamle store små
viktig huske husk bli være måten måte vanlig vanlige eksempel ting flere finne gang ganger trenger ønsker
hverandre personen man kanskje mer mindre ofte sjelden derfor altså helt selv kun bare omtrent slik
spørsmål svar svare nyttig hjelp helsestasjon gratis tilbud finner mer informasjon les les mer
""".split())

def tokens(s: str):
    return re.findall(r"[a-zæøåA-ZÆØÅ][a-zæøåA-ZÆØÅ\-]+", s.lower())

def freq(texts):
    c = Counter()
    for t in texts:
        for tok in tokens(t):
            if len(tok) >= 3 and tok not in STOP:
                c[tok] += 1
    return c

cluster_freq = {label: freq(texts) for label, texts in by_cluster.items()}
global_freq = Counter()
for c in cluster_freq.values():
    global_freq.update(c)

# Bigram frequency per cluster
def bigram_freq(texts):
    c = Counter()
    for t in texts:
        toks = tokens(t)
        for i in range(len(toks) - 1):
            a, b = toks[i], toks[i+1]
            if a in STOP or b in STOP: continue
            if len(a) < 3 or len(b) < 3: continue
            c[(a, b)] += 1
    return c

cluster_bigrams = {label: bigram_freq(texts) for label, texts in by_cluster.items()}
global_bigrams = Counter()
for c in cluster_bigrams.values():
    global_bigrams.update(c)

# Log-odds with Dirichlet smoothing — pick top distinctive terms per cluster
def discriminative(term_counts_in, term_counts_out, min_in=3, top_n=30):
    total_in  = sum(term_counts_in.values())   or 1
    total_out = sum(term_counts_out.values())  or 1
    alpha = 0.5
    scored = []
    for term, cin in term_counts_in.items():
        if cin < min_in:
            continue
        cout = term_counts_out.get(term, 0)
        # log-odds ratio
        p_in  = (cin + alpha)  / (total_in  + alpha * 2)
        p_out = (cout + alpha) / (total_out + alpha * 2)
        score = log(p_in / p_out) * cin   # weight by raw count
        scored.append((term, cin, cout, score))
    scored.sort(key=lambda x: x[3], reverse=True)
    return scored[:top_n]

print("=" * 80)
for label in [c[0] for c in CLUSTERS] + ["_Other_"]:
    if label not in cluster_freq:
        continue
    print(f"\n## {label}")
    print(f"   sources ({len(sources_per_cluster[label])}):")
    for s in sources_per_cluster[label][:6]:
        # Show only the slug tail
        tail = s.rsplit("/", 1)[-1].replace(".html", "")[:70]
        print(f"     - {tail}")
    if len(sources_per_cluster[label]) > 6:
        print(f"     ... +{len(sources_per_cluster[label]) - 6} more")

    # exclude this cluster's counts to compute "out"
    out_counts = Counter()
    for other_label, fc in cluster_freq.items():
        if other_label != label:
            out_counts.update(fc)
    print("   top unigrams:")
    for term, cin, cout, score in discriminative(cluster_freq[label], out_counts, min_in=4, top_n=25):
        print(f"     {cin:4d}/{cout:4d}  {term}")

    out_b = Counter()
    for other_label, fb in cluster_bigrams.items():
        if other_label != label:
            out_b.update(fb)
    print("   top bigrams:")
    for (a, b), cin, cout, score in [
        ((a, b), cin, cout, s) for ((a, b), cin, cout, s) in (
            (term, *rest) for term, *rest in discriminative(
                {k: v for k, v in cluster_bigrams[label].items()},
                {k: v for k, v in out_b.items()},
                min_in=3, top_n=15,
            )
        )
    ]:
        print(f"     {cin:4d}/{cout:4d}  {a} {b}")
