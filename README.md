# Complexity-Aware Adaptive Retrieval-Augmented Generation for Grounded Kazakh Legal Question Answering

This repository contains the code, configuration, and evaluation pipeline for the paper:

> **Complexity-Aware Adaptive Retrieval-Augmented Generation for Grounded Kazakh Legal Question Answering**
> A. Tleubayeva, A. Mansurova, A. Shomanov, Zh. Makhambetova, P. (Sarisaray) Boluk.
<img width="524" height="294" alt="image" src="https://github.com/user-attachments/assets/83bc20d7-6034-4181-80c5-65745116809e" />

It accompanies two released artifacts:

- **LegalRAG corpus** — a structure-aware Kazakh legal retrieval corpus of 56,916 chunks from 5,603 documents.
  Hugging Face: https://huggingface.co/datasets/Arailym-tleubayeva/LegalRAG
- **Kazakh Legal QA benchmark** — 242 questions (182 answerable, 60 unanswerable), complexity-annotated and expert-validated.

> **Note on the main finding.** This study reports a *controlled negative result*: complexity-aware
> dynamic-depth routing performs on par with a strong static hybrid baseline (Wilcoxon signed-rank,
> Holm-corrected p > 0.05 on all retrieval metrics). The productive levers are hybrid retrieval,
> second-stage reranking, and evidence grounding — not complexity-conditioned depth.

---

## 1. Repository structure

```
kaz-legal-rag/
├── README.md                     # this file
├── requirements.txt              # pinned dependencies
├── LICENSE                       # MIT (code) — see note on data licensing below
├── CITATION.cff                  # how to cite this work
├── .gitignore
│
├── notebooks/
│   ├── 01_corpus_construction.ipynb     # chunking + metadata (LegalRAG build)
│   ├── 02_retrieval_evaluation.ipynb    # dense backbone, hybrid, folding, reranking
│   ├── 03_systems_comparison.ipynb      # Naive / Static Hybrid / Adaptive + Wilcoxon
│   └── 04_generation_and_judge.ipynb    # generation, GPT-5 judge, abstention, human validation
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # global constants (seeds, k-values, model ids, kappa)
│   ├── retrieval.py              # BM25, BGE-M3, RRF fusion, reranker
│   ├── chunking.py               # structure-aware chunking
│   ├── prompts.py                # system + user prompts (matches paper Appendix A)
│   ├── generation.py             # generator backends (HF + OpenAI)
│   ├── judge.py                  # GPT-5 judge (6 criteria, 0–2) + GPT-4o cross-check
│   ├── abstention.py             # rule-based abstention detector (KZ/RUS)
│   ├── metrics.py                # retrieval + lexical/semantic metrics
│   └── stats.py                  # Wilcoxon+Holm, bootstrap CIs, Fleiss' kappa
│
├── configs/
│   └── experiment.yaml           # all hyperparameters in one place
│
├── data/
│   ├── README.md                 # how to obtain corpus + benchmark
│   └── benchmark/                # 242-question benchmark (or pointer to release)
│
└── results/
    ├── tables/                   # CSV of every table in the paper
    └── figures/                  # generated figures
```

> If your current repository keeps everything in one or two notebooks, that is fine — the
> `src/` layout is the recommended target. At minimum, ship `requirements.txt`, this `README`,
> the benchmark, and notebooks that run top-to-bottom.

---

## 2. Quick start

```bash
git clone https://github.com/Arailym-ray/kaz-legal-rag.git
cd kaz-legal-rag
python -m venv .venv && source .venv/bin/activate     # Python 3.10+
pip install -r requirements.txt
```

Download the corpus from Hugging Face:

```python
from datasets import load_dataset
corpus = load_dataset("Arailym-tleubayeva/LegalRAG")
```

Set API keys for the proprietary models (generation + judge):

```bash
export OPENAI_API_KEY="sk-..."     # GPT-4.1 generator, GPT-5 judge, GPT-4o cross-check
```

Then run the notebooks in `notebooks/` in order (01 → 04), or the scripted pipeline:

```bash
python -m src.run_all --config configs/experiment.yaml
```

---

## 3. Hardware and runtime

All experiments were run on a **single NVIDIA A100 GPU** (dense embedding and local LLM inference on
GPU; sparse retrieval and metric computation on CPU). Dense encoding used batch size 128 and a maximum
sequence length of 512 tokens. Generation and judging used deterministic decoding (temperature 0) with
a fixed random seed where applicable.

Approximate runtime on one A100: corpus indexing < 10 min; full retrieval evaluation ≈ 20–30 min;
generation + judge over the evaluation subset depends on API latency and budget.

---

## 4. Reproducing the paper's results

| Paper section | Notebook | Output |
|---|---|---|
| Corpus construction (3.2–3.3, Table 3–4) | `01_corpus_construction.ipynb` | LegalRAG chunks + stats |
| Dense backbone (4.3, Table 16) | `02_retrieval_evaluation.ipynb` | BGE-M3 selected |
| Lexical/hybrid (4.4, Table 17) | `02_…` | Static Hybrid best |
| Character folding (4.5, Table 18) | `02_…` | small +0.011 Hit@1 |
| Reranking ablation (4.6, Table 19) | `02_…` | top-20 best, +0.110 Hit@1 |
| Query analyzer (4.7, Table 20) | `03_systems_comparison.ipynb` | complexity acc 0.40 |
| Three systems + Wilcoxon (4.8, Table 21) | `03_…` | Static≈Adaptive (n.s.) |
| By complexity (4.9, Table 22) | `03_…` | selective benefit |
| Generator selection (4.10.1, Table 23) | `04_generation_and_judge.ipynb` | GPT-4.1 selected |
| End-to-end RAG (4.10.2, Table 24–25) | `04_…` | +0.80 legal correctness |
| Abstention (4.10.3, Table 26) | `04_…` | precision/recall/F1 |
| Judge validation (4.10.4, Table 27–28) | `04_…` | human vs GPT-5 |

Per-question scores are saved under `results/tables/` so that the significance tests
(Wilcoxon + Holm, bootstrap CIs) can be re-run without regenerating answers.

---

## 5. Key configuration (single source of truth)

| Component | Value |
|---|---|
| Random seed | `42` |
| Dense backbone | `BAAI/bge-m3` |
| Sparse retriever | BM25 (Okapi) |
| Hybrid fusion | Reciprocal Rank Fusion, κ = 60 |
| Reranker (ablation) | `BAAI/bge-reranker-v2-m3` |
| Retrieval depth | Naive k=5; Static k=10; Adaptive k∈{5,10,15} |
| Generator | `gpt-4.1` (selected via backbone comparison) |
| Generator candidates | KazLLM-1.0-8B, Hermes-3-Llama-3.1-8B, Qwen2.5-7B, GPT-4.1 |
| Primary judge | `gpt-5` (6 criteria, 0–2 ordinal) |
| Secondary judge | `gpt-4o` (cross-check) |
| Decoding | temperature 0, max_new_tokens 512 |
| FAISS index | `IndexFlatIP` on L2-normalised embeddings |

All of the above live in `configs/experiment.yaml` and `src/config.py`.

---

## 6. Data and licensing

- **Code**: released under the MIT License (see `LICENSE`).
- **LegalRAG corpus**: derived from the Kazakh legislation dataset of Rakhimova et al. (2025).
  See the dataset card on Hugging Face for licensing and reuse terms; cite the original source.
- **Benchmark**: released for research use. See `data/README.md`.

---

## 7. Citation

If you use this code, the corpus, or the benchmark, please cite the paper and the dataset
(see `CITATION.cff`). A BibTeX entry will be added once the paper DOI is assigned.

---

## 8. Limitations (mirrors the paper)

- The benchmark is moderate in size (242 questions) and single-gold at the chunk level.
- The generator (GPT-4.1) and primary judge (GPT-5) share a provider family; cross-checked with
  GPT-4o and three human legal experts. The automatic hallucination metric is the least reliable.
- The adaptive mechanism varies retrieval depth/scope only; richer adaptive behaviours are specified
  but not fully ablated.
- Corpus and benchmark are specific to Kazakhstani legislation.

---

## 9. Contact

Arailym Tleubayeva — a.tleubayeva@astanait.edu.kz
Astana IT University, Astana, Kazakhstan.
