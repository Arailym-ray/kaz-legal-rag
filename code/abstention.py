"""Rule-based abstention detector (Section 3.8.4).

Classifies a generated answer as an abstention if it contains explicit refusal /
insufficiency markers. Markers cover Kazakh and Russian (the two answer languages);
add English only if English refusals were actually produced in your run — keep this
in sync with the paper's Section 3.8.4 and configs/experiment.yaml.
"""
import re

# Explicit refusal / insufficiency markers used in the prompts (Appendix A)
ABSTENTION_MARKERS = {
    "kz": [
        "ЖАУАП ЖОҚ",
        "жауап жоқ",
        "мәлімет жеткіліксіз",
        "дереккөз жеткіліксіз",
        "анықтау мүмкін емес",
    ],
    "rus": [
        "НЕТ ОТВЕТА",
        "нет ответа",
        "недостаточно",
        "не может быть определ",        # определён / определено
        "невозможно определить",
    ],
    # "en": ["NO ANSWER", "insufficient evidence", "cannot be determined"],
}


def is_abstention(answer: str, languages=("kz", "rus")) -> bool:
    """Return True if the answer is an abstention.

    Matching is case-insensitive and substring-based, which is appropriate for the
    fixed refusal templates defined in the system prompts.
    """
    if not answer:
        return True  # empty / failed generation counts as a non-answer
    text = answer.lower()
    for lang in languages:
        for marker in ABSTENTION_MARKERS.get(lang, []):
            if marker.lower() in text:
                return True
    return False


def abstention_metrics(records, languages=("kz", "rus")):
    """Compute abstention precision/recall/F1 with unanswerable as positive class.

    records : iterable of dicts with keys:
        - "answerable": bool (True if the gold question is answerable)
        - "answer": str (the generated answer)

    Returns dict with counts and precision/recall/F1.
    """
    tp = fp = fn = tn = 0
    correct_abstain = false_abstain = 0
    for r in records:
        abstained = is_abstention(r["answer"], languages=languages)
        unanswerable = not r["answerable"]
        if abstained and unanswerable:
            tp += 1
            correct_abstain += 1
        elif abstained and not unanswerable:
            fp += 1
            false_abstain += 1
        elif not abstained and unanswerable:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "correct_abstention": correct_abstain,
        "false_abstention": false_abstain,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
