# Notebooks

Run in order. Each notebook imports configuration from `src/config.py` and writes
per-question scores to `results/tables/` so the statistics can be re-run later.

1. `01_corpus_construction.ipynb` — structure-aware chunking, metadata, LegalRAG stats (Tables 3–4).
2. `02_retrieval_evaluation.ipynb` — dense backbone (Table 16), lexical/hybrid (Table 17),
   character folding (Table 18), reranking ablation (Table 19).
3. `03_systems_comparison.ipynb` — query analyzer (Table 20), three systems + Wilcoxon (Table 21),
   by-complexity (Table 22).
4. `04_generation_and_judge.ipynb` — generator selection (Table 23), end-to-end RAG (Tables 24–25),
   abstention (Table 26), judge validation vs GPT-4o and human experts (Tables 27–28).

> Replace your existing single-file notebooks with these four, or keep your originals and add
> a top cell in each that `from src.config import *`. The goal is that a reviewer can open any
> notebook and see which paper table it produces.
