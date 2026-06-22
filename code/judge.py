"""LLM-as-a-judge evaluation (Section 3.7.5).

Primary judge: GPT-5. Six criteria, each scored 0-2 (ordinal). Hallucination is a
negative metric (0 = none, 1 = minor, 2 = major). Secondary judge GPT-4o is used
for a cross-family robustness check (Section 4.10.4).

Decoding is deterministic (temperature 0). The judge compares the system answer
against the question, the gold answer, and the supporting evidence.
"""
import json
from .config import JUDGE_PRIMARY, JUDGE_TEMPERATURE, JUDGE_CRITERIA

JUDGE_SYSTEM_PROMPT = (
    "You are a strict bilingual (Kazakh/Russian) legal-QA evaluator. "
    "Treat Kazakh morphological variants as equivalent. Score the SYSTEM ANSWER "
    "against the QUESTION, the GOLD ANSWER, and the SUPPORTING EVIDENCE on six "
    "criteria, each an integer 0, 1, or 2:\n"
    "- legal_correctness: legally valid w.r.t. the gold answer and evidence.\n"
    "- evidence_support: grounded in the supporting legal evidence.\n"
    "- citation_accuracy: cited law/article/clause matches the evidence.\n"
    "- completeness: includes necessary conditions, exceptions, qualifications.\n"
    "- clarity: understandable and well structured.\n"
    "- hallucination: unsupported/fabricated content (0=none, 1=minor, 2=major).\n"
    "Return ONLY a JSON object with these six integer fields and no other text."
)


def build_judge_user_prompt(question, gold_answer, evidence, system_answer):
    return (
        f"QUESTION:\n{question}\n\n"
        f"GOLD ANSWER:\n{gold_answer}\n\n"
        f"SUPPORTING EVIDENCE:\n{evidence}\n\n"
        f"SYSTEM ANSWER:\n{system_answer}\n\n"
        "Score the SYSTEM ANSWER. Respond with JSON only."
    )


def judge_answer(client, question, gold_answer, evidence, system_answer,
                 model=JUDGE_PRIMARY):
    """Call the judge model and parse its JSON scores.

    `client` is an OpenAI-compatible client. Returns dict of the six criteria,
    or None on parse failure (caller should log and skip).
    """
    resp = client.chat.completions.create(
        model=model,
        temperature=JUDGE_TEMPERATURE,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_user_prompt(
                question, gold_answer, evidence, system_answer)},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        return None
    # keep only the expected criteria, coerce to int
    return {c: int(scores[c]) for c in JUDGE_CRITERIA if c in scores}
