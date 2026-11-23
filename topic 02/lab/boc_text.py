"""Fetch Bank of Canada rate-announcement press releases and cache them as JSONL.

Used by Step 02 (hybrid retrieval). Run once on your Mac:
    python -m lab.boc_text
It writes data/boc_statements.jsonl  (one line per release: date, url, title, text).

The parser is deliberately simple: take the <h1> as title and the <p> paragraphs
inside the main content area. Check the printed word counts — a normal release
is roughly 300-700 words. If a page comes back with ~0 words, the site layout
changed; open the URL and adjust MAIN_SELECTORS below.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# Policy-rate announcements, 2025. Add more (e.g. 2026 dates) for "try next".
URLS = {
    "2025-01-29": "https://www.bankofcanada.ca/2025/01/fad-press-release-2025-01-29/",
    "2025-03-12": "https://www.bankofcanada.ca/2025/03/fad-press-release-2025-03-12/",
    "2025-04-16": "https://www.bankofcanada.ca/2025/04/fad-press-release-2025-04-16/",
    "2025-09-17": "https://www.bankofcanada.ca/2025/09/fad-press-release-2025-09-17/",
    "2025-10-29": "https://www.bankofcanada.ca/2025/10/fad-press-release-2025-10-29/",
    "2025-12-10": "https://www.bankofcanada.ca/2025/12/fad-press-release-2025-12-10/",
}

MAIN_SELECTORS = ["#main-content", "main", "article", "body"]
# Boilerplate lines we don't want in the corpus
SKIP_STARTS = ("Media Relations", "Information note", "Content Type", "FOR IMMEDIATE RELEASE")

OUT = Path(__file__).resolve().parents[1] / "data" / "boc_statements.jsonl"


def parse_release(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else ""
    main = None
    for sel in MAIN_SELECTORS:
        main = soup.select_one(sel)
        if main is not None:
            break
    paras = []
    for p in main.find_all("p"):
        t = p.get_text(" ", strip=True)
        if len(t.split()) < 8 or t.startswith(SKIP_STARTS):
            continue
        paras.append(t)
    return title, "\n".join(paras)


def fetch_all(urls: dict[str, str] = URLS, out: Path = OUT) -> list[dict]:
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    headers = {"User-Agent": "ai-econ-lab (personal research project)"}
    for date, url in urls.items():
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  ! skip {date}: {e}")
            continue
        title, text = parse_release(r.text)
        print(f"  {date}  {len(text.split()):4d} words  {title[:70]}")
        rows.append({"date": date, "url": url, "title": title, "text": text})
        time.sleep(1)  # be polite
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} releases -> {out}")
    return rows


def load_statements(path: Path = OUT) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m lab.boc_text` first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    fetch_all()
