"""
Retrieval-quality eval for query_project_docs_tool's RAG pipeline - not a
pytest test (not named test_*.py on purpose, so pytest's default
`testpaths = tests` collection in pytest.ini skips it; it needs the real
embedding model and a built Chroma index, neither of which belong in the
fast, fully-mocked unit suite). Run manually after rebuilding the
knowledge base:

    python src/agent/build_knowledge_base.py
    python tests/eval/rag_eval.py               # vector-only baseline
    python tests/eval/rag_eval.py --hybrid       # BM25+vector hybrid retriever

Runs a fixed set of 17 question/expected-source pairs (6 exact-term - the
query uses the same wording/numbers/policy IDs as the source doc; 8
semantic paraphrase - same underlying fact, deliberately different
wording, no exact term overlap; 3 unanswerable - no doc covers it)
against either retriever, reports hit@1/hit@3 per category, and flags
any unanswerable question whose top retrieved chunk scores above
SUSPICIOUS_SCORE_THRESHOLD - a high score on a question nothing should
answer is a sign the retriever would confidently hand the agent an
irrelevant chunk instead of coming back empty. Saves the full result to
reports/rag_eval_baseline.json (vector) or reports/rag_eval_hybrid.json
(--hybrid).

The 0.5 threshold was chosen empirically against the vector-only
baseline: a local run scored the 6 exact-term questions' top hits at
0.70-0.90 relevance and the 3 unanswerable questions' top hits at
0.27-0.31 - a wide gap with plenty of margin either side, not a number
tuned to make this pass. The hybrid retriever's score isn't the same
underlying quantity (see src/agent/hybrid_retriever.py's
hybrid_search_with_scores - a fused-RRF score normalized by the maximum
possible RRF score, not a cosine similarity), but it's normalized onto
the same [0, 1]-ish "fraction of this retriever's own max confidence"
scale, so the same 0.5 threshold is still a meaningful, comparable check
even though the two numbers aren't the same thing underneath.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

# Run directly as `python tests/eval/rag_eval.py` (not via pytest, which
# gets pythonpath=. from pytest.ini) - put the project root on sys.path
# first, same fallback pattern src/agent/tools.py uses.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agent.hybrid_retriever import hybrid_search_with_scores  # noqa: E402
from src.agent.tools import _get_vectorstore  # noqa: E402
from src.common.embedding_model import EMBEDDING_MODEL_NAME, get_embedder, set_embedder  # noqa: E402

BASELINE_REPORT_PATH = Path("reports/rag_eval_baseline.json")
HYBRID_REPORT_PATH = Path("reports/rag_eval_hybrid.json")
TOP_K = 3
SUSPICIOUS_SCORE_THRESHOLD = 0.5

# expected_sources is a list because a couple of facts are legitimately
# covered in more than one doc (e.g. num_addons_active is explained in
# both the glossary and the data dictionary) - a hit against ANY listed
# source counts, matching how a real user would be satisfied by either.
QUESTIONS = [
    # --- exact-term (6): query reuses the doc's own wording/numbers/IDs ---
    {
        "question": "What is the early termination fee structure under policy POL-CTR-009?",
        "expected_sources": ["contract_terms.md"],
        "category": "exact_term",
    },
    {
        "question": "What promo code gives $200 off for new fiber customers?",
        "expected_sources": ["fiber_pricing_promotions.md"],
        "category": "exact_term",
    },
    {
        "question": "How many calendar days is the cancellation window for a full refund?",
        "expected_sources": ["refund_cancellation_policy.md"],
        "category": "exact_term",
    },
    {
        "question": "What is the monthly uptime guarantee percentage in the support SLA?",
        "expected_sources": ["support_sla.md"],
        "category": "exact_term",
    },
    {
        "question": "What is the overall churn rate in the training dataset?",
        "expected_sources": ["glossary.md", "model_card_churn_xgboost.md"],
        "category": "exact_term",
    },
    {
        "question": "What is XGBoost's PR-AUC score for the churn model?",
        "expected_sources": ["model_card_churn_xgboost.md", "glossary.md"],
        "category": "exact_term",
    },
    # --- semantic paraphrase (8): same fact, deliberately different wording ---
    {
        "question": "If I cancel my 2-year plan early, how much extra will I be charged?",
        "expected_sources": ["contract_terms.md"],
        "category": "paraphrase",
    },
    {
        "question": "I just signed up - can I get my money back if I change my mind this week?",
        "expected_sources": ["refund_cancellation_policy.md"],
        "category": "paraphrase",
    },
    {
        "question": "Is there a discount if I pay through my bank account automatically each month?",
        "expected_sources": ["autopay_payment_policy.md"],
        "category": "paraphrase",
    },
    {
        "question": "What happens to my bill if the internet goes down for a long time?",
        "expected_sources": ["outage_credit_policy.md"],
        "category": "paraphrase",
    },
    {
        "question": "My customer keeps saying it's too expensive and wants to leave - what should I offer them?",
        "expected_sources": ["retention_playbook_price_sensitive.md"],
        "category": "paraphrase",
    },
    {
        "question": "How accurate is the churn prediction model and what are its known weaknesses?",
        "expected_sources": ["model_card_churn_xgboost.md"],
        "category": "paraphrase",
    },
    {
        "question": "Why do customers who've been with the company for less time leave more often?",
        "expected_sources": ["glossary.md", "decisions.md"],
        "category": "paraphrase",
    },
    {
        "question": "Why did the team pick a different scoring metric than plain accuracy for the churn model?",
        "expected_sources": ["glossary.md"],
        "category": "paraphrase",
    },
    # --- unanswerable (3): no doc covers this ---
    {
        "question": "What is the CEO's favorite color?",
        "expected_sources": None,
        "category": "unanswerable",
    },
    {
        "question": "Does Vantrix offer satellite internet service in rural areas?",
        "expected_sources": None,
        "category": "unanswerable",
    },
    {
        "question": "What is the customer service phone number for the UK office?",
        "expected_sources": None,
        "category": "unanswerable",
    },
]


def _source_basename(metadata):
    return Path(metadata.get("source", "unknown")).name


def evaluate_question(vectorstore, item, hybrid):
    if hybrid:
        results = hybrid_search_with_scores(vectorstore, get_embedder(), item["question"], k=TOP_K)
    else:
        results = vectorstore.similarity_search_with_relevance_scores(item["question"], k=TOP_K)
    retrieved = [
        {"source": _source_basename(doc.metadata), "section": doc.metadata.get("section"), "score": round(score, 4)}
        for doc, score in results
    ]
    retrieved_sources = [r["source"] for r in retrieved]
    top_score = retrieved[0]["score"] if retrieved else 0.0

    record = {
        "question": item["question"],
        "category": item["category"],
        "expected_sources": item["expected_sources"],
        "retrieved": retrieved,
        "top_score": top_score,
    }

    if item["category"] == "unanswerable":
        record["suspicious"] = top_score > SUSPICIOUS_SCORE_THRESHOLD
    else:
        expected = set(item["expected_sources"])
        record["hit_at_1"] = retrieved_sources[0] in expected if retrieved_sources else False
        record["hit_at_3"] = any(s in expected for s in retrieved_sources)

    return record


def summarize(records):
    answerable = [r for r in records if r["category"] != "unanswerable"]
    unanswerable = [r for r in records if r["category"] == "unanswerable"]

    def rate(items, key):
        return round(sum(1 for r in items if r[key]) / len(items), 4) if items else None

    by_category = {}
    for category in ("exact_term", "paraphrase"):
        items = [r for r in answerable if r["category"] == category]
        by_category[category] = {
            "n": len(items),
            "hit_at_1": rate(items, "hit_at_1"),
            "hit_at_3": rate(items, "hit_at_3"),
        }

    return {
        "overall_hit_at_1": rate(answerable, "hit_at_1"),
        "overall_hit_at_3": rate(answerable, "hit_at_3"),
        "by_category": by_category,
        "unanswerable": {
            "n": len(unanswerable),
            "suspicious_count": sum(1 for r in unanswerable if r["suspicious"]),
            "suspicious_threshold": SUSPICIOUS_SCORE_THRESHOLD,
            "top_scores": [r["top_score"] for r in unanswerable],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hybrid", action="store_true", help="Evaluate the BM25+vector hybrid retriever instead of vector-only."
    )
    args = parser.parse_args()

    set_embedder(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
    vectorstore = _get_vectorstore()

    records = [evaluate_question(vectorstore, item, args.hybrid) for item in QUESTIONS]
    summary = summarize(records)

    report_path = HYBRID_REPORT_PATH if args.hybrid else BASELINE_REPORT_PATH
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "retriever": "hybrid_bm25_vector_rrf" if args.hybrid else "vector_only",
        "top_k": TOP_K,
        "n_questions": len(QUESTIONS),
        "summary": summary,
        "results": records,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nretriever: {report['retriever']}")
    print(f"{'category':<12}{'n':>4}{'hit@1':>10}{'hit@3':>10}")
    for category, stats in summary["by_category"].items():
        print(f"{category:<12}{stats['n']:>4}{stats['hit_at_1']:>10}{stats['hit_at_3']:>10}")
    print(f"{'overall':<12}{len(QUESTIONS) - summary['unanswerable']['n']:>4}"
          f"{summary['overall_hit_at_1']:>10}{summary['overall_hit_at_3']:>10}")

    u = summary["unanswerable"]
    print(f"\nunanswerable: {u['n']} questions, top scores {u['top_scores']}, "
          f"{u['suspicious_count']} above suspicious threshold ({u['suspicious_threshold']})")
    print(f"\nSaved full report to {report_path}")


if __name__ == "__main__":
    main()
