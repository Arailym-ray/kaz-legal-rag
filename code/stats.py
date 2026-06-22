"""Statistical tests used in the paper (Section 3.8.6).

- Paired Wilcoxon signed-rank with Holm correction (retrieval metrics, Table 21).
- Paired bootstrap confidence intervals (generation deltas, Table 25).
- Fleiss' kappa (inter-annotator agreement, Table 27).
"""
import numpy as np
from scipy.stats import wilcoxon


def wilcoxon_holm(per_question_a, per_question_b, metric_names):
    """Paired Wilcoxon signed-rank with Holm correction across metrics.

    per_question_a, per_question_b : dict[metric_name -> np.ndarray of per-question scores]
    Returns dict[metric_name -> (statistic, p_raw, p_holm, significant)].
    """
    raw = {}
    for m in metric_names:
        a = np.asarray(per_question_a[m], dtype=float)
        b = np.asarray(per_question_b[m], dtype=float)
        if np.allclose(a, b):
            raw[m] = (0.0, 1.0)
            continue
        stat, p = wilcoxon(a, b)
        raw[m] = (stat, p)

    # Holm correction across the metric family
    items = sorted(raw.items(), key=lambda kv: kv[1][1])  # by raw p ascending
    n = len(items)
    out = {}
    prev = 0.0
    for i, (m, (stat, p)) in enumerate(items):
        p_holm = min(1.0, max(prev, (n - i) * p))
        prev = p_holm
        out[m] = {
            "statistic": stat,
            "p_raw": p,
            "p_holm": p_holm,
            "significant": p_holm < 0.05,
        }
    return out


def paired_bootstrap_ci(diff, n_iter=10_000, ci=0.95, seed=42):
    """Bootstrap CI for the mean of a paired difference vector (RAG - LLM-only).

    diff : 1D array of per-question differences.
    Returns (mean, lo, hi, significant) where significant means the CI excludes 0.
    """
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    means = np.empty(n_iter)
    for i in range(n_iter):
        sample = diff[rng.integers(0, n, n)]
        means[i] = sample.mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    mean = diff.mean()
    significant = (lo > 0) or (hi < 0)
    return round(float(mean), 3), round(float(lo), 3), round(float(hi), 3), bool(significant)


def fleiss_kappa(table):
    """Fleiss' kappa for inter-annotator agreement.

    table : np.ndarray of shape (n_items, n_categories), entry = number of raters
            who assigned that category to that item. Row sums must be equal (= n_raters).
    """
    table = np.asarray(table, dtype=float)
    n_items, n_cat = table.shape
    n_raters = table[0].sum()
    p_j = table.sum(axis=0) / (n_items * n_raters)            # category proportions
    P_i = (np.square(table).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e = np.square(p_j).sum()
    if np.isclose(1 - P_e, 0):
        return float("nan")                                   # kappa paradox guard
    return round(float((P_bar - P_e) / (1 - P_e)), 3)
