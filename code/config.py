"""Global configuration for the Kazakh Legal RAG experiments.

Values here mirror configs/experiment.yaml and the paper (Tables 9, 10).
Importing from one place prevents the generator/judge/seed drifting between
notebooks — a common source of irreproducibility.
"""
from pathlib import Path

# ----- reproducibility -----
RANDOM_SEED = 42

# ----- corpus -----
CORPUS_HF_ID = "Arailym-tleubayeva/LegalRAG"
N_CHUNKS = 56_916
N_DOCUMENTS = 5_603
MIN_CHUNK_WORDS = 30
MAX_CHUNK_WORDS = 450

# ----- retrieval -----
DENSE_BACKBONE = "BAAI/bge-m3"
DENSE_CANDIDATES = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large",
    "Qwen/Qwen3-Embedding-0.6B",
    "sentence-transformers/LaBSE",
]
RRF_KAPPA = 60                 # reciprocal rank fusion smoothing constant
FAISS_INDEX = "IndexFlatIP"
NORMALIZE_EMBEDDINGS = True
MAX_SEQ_LENGTH = 512
DENSE_BATCH_SIZE = 128         # fits comfortably on a single A100 (40/80 GB)
CHARACTER_FOLDING = True

# ----- reranker (ablation only) -----
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_POOLS = [20, 50, 100]
RERANKER_FINAL_TOP_K = 10

# ----- retrieval depth (Table 9) -----
NAIVE_K = 5
STATIC_K = 10
ADAPTIVE_K = {"simple": 5, "moderate": 10, "complex": 15}

# ----- generation -----
GENERATOR = "gpt-4.1"          # selected via backbone comparison (Table 23)
GENERATOR_CANDIDATES = [
    ("KazLLM-1.0-8B",         "issai/LLama-3.1-KazLLM-1.0-8B",      "hf"),
    ("Hermes-3-Llama-3.1-8B", "NousResearch/Hermes-3-Llama-3.1-8B", "hf"),
    ("Qwen2.5-7B",            "Qwen/Qwen2.5-7B-Instruct",           "hf"),
    ("GPT-4.1",               "gpt-4.1",                            "openai"),
]
GEN_TEMPERATURE = 0
GEN_MAX_NEW_TOKENS = 512
GEN_EVAL_N = 50                # stratified by complexity

# ----- judge -----
JUDGE_PRIMARY = "gpt-5"
JUDGE_SECONDARY = "gpt-4o"
JUDGE_TEMPERATURE = 0
JUDGE_CRITERIA = [
    "legal_correctness", "evidence_support", "citation_accuracy",
    "completeness", "clarity", "hallucination",
]

# ----- abstention -----
ABSTENTION_LANGUAGES = ["kz", "rus"]   # add "en" only if English markers were used

# ----- statistics -----
SIGNIFICANCE_TEST = "wilcoxon"
CORRECTION = "holm"
ALPHA = 0.05
BOOTSTRAP_ITERS = 10_000
CI = 0.95

# ----- paths -----
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
BENCHMARK_PATH = ROOT / "data" / "benchmark" / "benchmark_242.jsonl"
