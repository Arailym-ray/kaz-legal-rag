"""Evaluation metrics (Section 3.8).

Retrieval: Hit@1 (=Recall@1 single-gold), Recall@K, MRR@10, nDCG@10,
           DocRecall@10, SourceRecall@10.
Answer:    Exact Match, Token-F1, ROUGE-L, chrF++, semantic similarity (BGE-M3 cosine).
Legal-grounding metrics are produced by the LLM judge (see judge.py), not here.
"""
import numpy as np


# --------------------------- retrieval ---------------------------

def hit_at_1(ranked_ids, gold_id):
    return 1.0 if ranked_ids and ranked_ids[0] == gold_id else 0.0


def recall_at_k(ranked_ids, gold_id, k):
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def mrr(ranked_ids, gold_id):
    for i, cid in enumerate(ranked_ids):
        if cid == gold_id:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_10(ranked_ids, gold_id):
    for i, cid in enumerate(ranked_ids[:10]):
        if cid == gold_id:
            return 1.0 / np.log2(i + 2)   # IDCG = 1 for single gold
    return 0.0


def retrieval_summary(results, gold_ids, ks=(1, 3, 5, 10)):
    """results : list[list[chunk_id]] best-first; gold_ids : list[chunk_id]."""
    out = {"Hit@1": [], "MRR@10": [], "nDCG@10": []}
    for k in ks:
        out[f"Recall@{k}"] = []
    for ranked, gold in zip(results, gold_ids):
        out["Hit@1"].append(hit_at_1(ranked, gold))
        out["MRR@10"].append(mrr(ranked[:10], gold))
        out["nDCG@10"].append(ndcg_at_10(ranked, gold))
        for k in ks:
            out[f"Recall@{k}"].append(recall_at_k(ranked, gold, k))
    return {m: float(np.mean(v)) for m, v in out.items()}, out  # means, per-question


# --------------------------- answer similarity ---------------------------

def exact_match(pred, gold):
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


def token_f1(pred, gold):
    p, g = pred.lower().split(), gold.lower().split()
    if not p or not g:
        return 0.0
    common = {}
    for t in p:
        common[t] = min(p.count(t), g.count(t))
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p)
    recall = n_same / len(g)
    return 2 * precision * recall / (precision + recall)


def cosine_sim(vec_a, vec_b):
    """Semantic similarity = cosine of L2-normalised BGE-M3 embeddings (Eq. 14)."""
    a, b = np.asarray(vec_a), np.asarray(vec_b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom else 0.0

# For chrF++ use sacrebleu:
#   from sacrebleu import sentence_chrf
#   sentence_chrf(pred, [gold], word_order=2).score
# For ROUGE-L use rouge_score.rouge_scorer with "rougeL".
