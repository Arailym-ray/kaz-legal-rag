# Complexity-Aware Adaptive RAG for Grounded Kazakh Legal Question Answering

This repository contains the data, code, and configuration for the paper
**"Complexity-Aware Adaptive Retrieval-Augmented Generation for Grounded
Kazakh Legal Question Answering"** (Tleubayeva et al., 2026).
![Uploading image.png…]()

## Overview

We study whether query complexity can serve as a control signal for adaptive
RAG in Kazakh legal QA. We release a structure-aware legal corpus, a
complexity-annotated benchmark, and a controlled evaluation of three
retrieve-then-generate systems (Naive, Static Hybrid, Adaptive).

**Key findings:**
- Hybrid retrieval (BM25 + BGE-M3) significantly improves evidence coverage
  over lexical or dense retrieval alone.
- Complexity-aware dynamic-depth routing performs **on par** with the static
  hybrid baseline — query complexity alone is an insufficient control signal.
- Second-stage reranking is the strongest retrieval lever (Hit@1 0.451 → 0.560).
- Evidence-grounded RAG raises legal correctness 2.5× over non-retrieval
  generation (0.54 → 1.35 on a 0–2 scale).

## Resources

- **Corpus (56,916 chunks):** https://huggingface.co/datasets/Arailym-tleubayeva/LegalRAG
- **Benchmark (242 questions):** `data/legalrag_qa_benchmark.csv`

## Repository structure

| Path | Description |
|------|-------------|
| `data/` | Benchmark (242 Q) and corpus pointer |
| `notebooks/` | End-to-end retrieval, generation, statistics |
| `src/` | Reusable modules (prompts, retrieval, generation, evaluation) |
| `results/` | Tables and figures from the paper |

## Installation

```bash
git clone https://github.com/Arailym-ray/KAZ-LEGAL-RAG.git
cd KAZ-LEGAL-RAG
pip install -r requirements.txt
```

Set API keys (for GPT-4.1 generation and GPT-5 judge):
```bash
export OPENAI_API_KEY=...
export HF_TOKEN=...            # for gated KazLLM
```

## Reproducing the results

1. **Retrieval** — `notebooks/01_retrieval_pipeline.ipynb`
   BM25, BGE-M3, hybrid (RRF), reranking; produces Tables 16–22.
2. **Generation** — `notebooks/02_generation_pipeline.ipynb`
   Backbone selection, RAG vs LLM-only, GPT-5 judge; Tables 23–28.
3. **Statistics & figures** — `notebooks/03_statistics_and_figures.ipynb`
   Bootstrap CIs, Wilcoxon + effect sizes, Figures 2–6.

## Models

| Role | Model |
|------|-------|
| Dense retriever | BGE-M3 |
| Reranker | BGE-reranker-v2-m3 |
| Generator | GPT-4.1 (selected via backbone comparison) |
| Judge | GPT-5 (validated against GPT-4o + 3 legal experts) |
| Kazakh baseline | issai/LLama-3.1-KazLLM-1.0-8B |

## Citation

```bibtex
@article{tleubayeva2026kazlegalrag,
  title   = {Complexity-Aware Adaptive Retrieval-Augmented Generation for
             Grounded Kazakh Legal Question Answering},
  author  = {Tleubayeva, Arailym and Mansurova, Aigerim and Shomanov, Aday
             and Boluk, Pinar},
  journal = {...},
  year    = {2026}
}
```

## License

Code: MIT. Benchmark and corpus: CC BY 4.0.
Note: KazLLM-1.0-8B is released under CC BY-NC 4.0 (non-commercial).
