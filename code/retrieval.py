"""Retrieval components: BM25, dense (BGE-M3), reciprocal rank fusion, reranker.

The fusion implements paper Equation (4):

    RRF(d) = sum_m 1 / (kappa + rank_m(d)),    kappa = 60

This is a reference implementation aligned with the paper; adapt the index/model
loading to your environment.
"""
from collections import defaultdict
from .config import RRF_KAPPA, STATIC_K


def reciprocal_rank_fusion(ranked_lists, kappa: int = RRF_KAPPA, top_k: int = STATIC_K):
    """Fuse several ranked lists of chunk ids with RRF.

    Parameters
    ----------
    ranked_lists : list[list[chunk_id]]
        Each inner list is ordered best-first (rank 1 = position 0).
    kappa : int
        Smoothing constant (paper uses 60).
    top_k : int
        Number of fused results to return.

    Returns
    -------
    list[(chunk_id, score)] ordered by descending fused score, deduplicated.
    """
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for position, chunk_id in enumerate(ranked):
            rank = position + 1          # 1-based rank
            scores[chunk_id] += 1.0 / (kappa + rank)
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused[:top_k]


def hybrid_retrieve(query, bm25_search, dense_search, kappa=RRF_KAPPA, top_k=STATIC_K, pool=50):
    """Static Hybrid retrieval: fuse BM25 and dense BGE-M3 rankings via RRF.

    `bm25_search` and `dense_search` are callables: query -> list[chunk_id]
    (best-first), each returning at least `pool` candidates.
    """
    sparse_ranked = bm25_search(query, top_k=pool)
    dense_ranked = dense_search(query, top_k=pool)
    return reciprocal_rank_fusion([sparse_ranked, dense_ranked], kappa=kappa, top_k=top_k)


def adaptive_retrieve(query, complexity, bm25_search, dense_search,
                      adaptive_k, kappa=RRF_KAPPA, pool=50):
    """Adaptive retrieval: depth conditioned on predicted complexity (Table 9).

    adaptive_k : dict like {"simple": 5, "moderate": 10, "complex": 15}
    """
    k = adaptive_k.get(complexity, adaptive_k["moderate"])
    return hybrid_retrieve(query, bm25_search, dense_search, kappa=kappa, top_k=k, pool=pool)
