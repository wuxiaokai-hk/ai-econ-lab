## Step 02 — Hybrid retrieval over Bank of Canada rate announcements (the "R" in RAG)

**What it shows.** Before an LLM can answer "when did the Bank end quantitative tightening?", a retriever has to find the right passage. This step builds three retrievers over the 2025 Bank of Canada rate-announcement press releases and scores them on a hand-labelled gold set:

| Retriever | Idea | Good at |
|---|---|---|
| BM25 | lexical score, implemented from the formula | exact terms and numbers ("2.25%", "quantitative tightening") |
| Dense | cosine similarity of local embeddings (`embeddinggemma` via Ollama) | paraphrases ("hold" vs "maintains") |
| Hybrid | Reciprocal Rank Fusion: `RRF(d) = Σ 1/(60 + rank)` | robust when either one misses |

Metrics: Recall@1, Recall@3 and MRR at the document level.

**Files.** `lab/boc_text.py` (fetch + cache releases) · `step02/step02_retrieval.py` (starter, 4 TODOs) · `step02/check_step02.py` (tests) · `step02/solution_step02_retrieval.py` (reference) · `data/step02_gold.csv` (6 questions)

**Run.**
```bash
pip install requests beautifulsoup4 numpy scikit-learn
python -m lab.boc_text                      # download + cache 6 releases
ollama pull embeddinggemma
python step02/check_step02.py               # until all PASS
python step02/step02_retrieval.py --embedder ollama
```

**What I learned.**
- BM25 is a formula: IDF × a saturating, length-normalised term frequency. You can check it by hand.
- Hard negatives matter. Three releases contain "2.25%" (the Bank Rate is 2.5% when the policy rate is 2.25%), so a number match alone does not find the right document.
- Lexical search needs normalisation: titles write "2¼%" and the body writes "2.25%", and "hold", "holds" and "held" count as different tokens.
- RRF fuses rankings, not scores, so BM25 and cosine never need to be put on the same scale.
- In RAG, retrieval recall caps answer quality. It is the first thing to validate (OSFI E-23 / SR 11-7 style).

*My results:* (paste the table here)

**Try next.** Add the 2026 announcements to `URLS`, write 5 meaning-only questions that share no words with the text (e.g. "When did the Bank warn of a recession from a trade war?"), and see whether dense or hybrid pulls ahead of BM25.
