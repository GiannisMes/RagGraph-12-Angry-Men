# RAG & GraphRAG over the *12 Angry Men* Screenplay

Two retrieval-augmented generation systems built over the screenplay of the film
*12 Angry Men*, then compared: a **traditional hybrid RAG** pipeline and a
**graph-based (GraphRAG)** pipeline. Both answer questions with
**Qwen3-4B-Instruct**. Built as a two-part AI lab project (NTUA, School of ECE).

**Authors:** Giannis Mertziotis · Spyridon Tzokas

---

## What's inside

| Folder | System | Retrieval | Generation |
| --- | --- | --- | --- |
| [`traditional-rag/`](traditional-rag/) | Hybrid RAG | BM25 (keyword) + FAISS (vector) over sentence-transformer embeddings | Qwen3-4B-Instruct |
| [`graphrag/`](graphrag/) | GraphRAG | k-hop traversal over a NetworkX knowledge graph + optional community summaries | Qwen3-4B-Instruct |

Each folder contains the notebook and its evaluation CSV(s).

## Traditional RAG (Part 2)

The screenplay PDF is downloaded at runtime, parsed with PyMuPDF, and split into
chunks. Each chunk is indexed twice — a **BM25** keyword index and a **FAISS**
vector index over sentence-transformer embeddings. At query time both indexes are
searched, the retrieved chunks become context for the LLM, and answers are written
to `evaluation.csv`.

## GraphRAG (Part 3)

Starting from a **pre-built knowledge graph** (`graph.json`) of entities and
relationships extracted from the screenplay, the system locates relevant nodes,
expands around them with **k-hop graph traversal** (NetworkX), serializes the
resulting subgraph into text evidence, optionally adds GraphRAG **community
summaries**, and passes the evidence to the LLM. Two runs are evaluated —
[`evaluation_with_communities.csv`](graphrag/evaluation_with_communities.csv) and
[`evaluation_without_communities.csv`](graphrag/evaluation_without_communities.csv) —
to measure the contribution of the summaries.

## Tech stack

`Python` · `Transformers` / `Qwen3-4B-Instruct` · `sentence-transformers` ·
`FAISS` · `rank_bm25` · `NetworkX` · `PyMuPDF` · `pandas`

## Running

The notebooks were developed in Google Colab (GPU recommended for the LLM).
Open a notebook, run the dependency-install cell, then run top to bottom.

## A note on data

The screenplay text and the extracted graph files are **not** committed to this
repository. The traditional-RAG notebook downloads the script at runtime; the
GraphRAG notebook expects the provided `graph.json`, `community_summaries.json`,
and `questions.csv` alongside it.
