# evaluator.py
# Measures search engine accuracy using Precision and Recall.
# You define the "ground truth" (relevant docs) for each test query,
# then run the engine and compare.

from searcher import search
from tfidf    import rank


# ═════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════

def precision(retrieved, relevant):
    """
    Precision = |Retrieved ∩ Relevant| / |Retrieved|

    "Of the docs we returned, how many are actually correct?"

    Edge cases handled:
    - Empty retrieved set → return 0.0 (cannot divide by zero)
    - Empty relevant set  → return 0.0 (nothing is correct)
    """
    if not retrieved:
        return 0.0
    if not relevant:
        return 0.0

    retrieved = set(retrieved)
    relevant  = set(relevant)

    return len(retrieved & relevant) / len(retrieved)


def recall(retrieved, relevant):
    """
    Recall = |Retrieved ∩ Relevant| / |Relevant|

    "Of all the correct docs that exist, how many did we find?"

    Edge cases handled:
    - Empty relevant set  → return 0.0 (nothing to recall)
    - Empty retrieved set → return 0.0
    """
    if not relevant:
        return 0.0
    if not retrieved:
        return 0.0

    retrieved = set(retrieved)
    relevant  = set(relevant)

    return len(retrieved & relevant) / len(relevant)


def f1_score(p, r):
    """
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Harmonic mean of precision and recall.

    Edge cases handled:
    - Both p and r are 0 → return 0.0
    """
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ═════════════════════════════════════════════════════════════
# GROUND TRUTH
# ═════════════════════════════════════════════════════════════

# Define your test queries and the doc_ids that are genuinely relevant.
# You must judge these manually by reading the corpus documents.
#
# FORMAT:
#   {
#     "query string": {
#         "relevant": ["doc_id_1", "doc_id_2", ...],
#         "language": "english" | "arabic" | None (auto-detect)
#     }
#   }
#
# Update this dict after you build your corpus and inspect the documents.

GROUND_TRUTH = {
    "climate change": {
        "relevant": ["doc_en_01.txt", "doc_en_10.txt"],
        "language": "english",
    },
    "artificial intelligence": {
        "relevant": ["doc_en_02.txt"],
        "language": "english",
    },
    "economy": {
        "relevant": ["doc_en_03.txt"],
        "language": "english",
    },
    "health": {
        "relevant": ["doc_en_05.txt"],
        "language": "english",
    },
    # Arabic query — at least one required by the project spec
    "تغير المناخ": {
        "relevant": ["doc_ar_01.txt", "doc_ar_10.txt"],
        "language": "arabic",
    },
}


# ═════════════════════════════════════════════════════════════
# EVALUATION RUNNER
# ═════════════════════════════════════════════════════════════

def evaluate(index, ground_truth=None, top_n=10):
    """
    Run every test query, compute Precision & Recall, and print a report.

    Parameters
    ----------
    index        : dict  – from indexer.get_index()
    ground_truth : dict  – custom ground truth (defaults to GROUND_TRUTH above)
    top_n        : int   – how many ranked results to consider

    Returns
    -------
    list of dicts, one per query, with keys:
        query, retrieved, relevant, precision, recall, f1

    Edge cases handled:
    - Empty ground truth        → warn and return []
    - Query returns no results  → precision=0, recall=0
    - Ground truth has no relevant docs → skip that query with a warning
    """
    if ground_truth is None:
        ground_truth = GROUND_TRUTH

    if not ground_truth:
        print("[EVAL] Ground truth is empty. Nothing to evaluate.")
        return []

    results = []

    print("\n" + "=" * 65)
    print("  Search Engine Evaluation Report")
    print("=" * 65)

    for query_str, meta in ground_truth.items():
        relevant = set(meta.get("relevant", []))
        language = meta.get("language", None)

        if not relevant:
            print(f"[EVAL] WARNING: No relevant docs defined for '{query_str}'. Skipping.")
            continue

        # Run the search
        search_result = search(query_str, index, language=language)
        ranked        = rank(
            search_result["query_tokens"],
            search_result["matches"],
            index,
            top_n=top_n,
        )
        retrieved = set(doc_id for doc_id, _ in ranked)

        # Compute metrics
        p  = precision(retrieved, relevant)
        r  = recall(retrieved, relevant)
        f1 = f1_score(p, r)

        results.append({
            "query":     query_str,
            "retrieved": retrieved,
            "relevant":  relevant,
            "precision": p,
            "recall":    r,
            "f1":        f1,
        })

        # ── Print per-query report ─────────────────────────────
        print(f"\n  Query     : {query_str}")
        print(f"  Language  : {search_result['language']}")
        print(f"  Retrieved : {sorted(retrieved) if retrieved else '(none)'}")
        print(f"  Relevant  : {sorted(relevant)}")
        print(f"  Precision : {p:.3f}")
        print(f"  Recall    : {r:.3f}")
        print(f"  F1 Score  : {f1:.3f}")

        if search_result["missing_terms"]:
            print(f"  OOV terms : {search_result['missing_terms']}")

    # ── Macro averages ─────────────────────────────────────────
    if results:
        avg_p  = sum(r["precision"] for r in results) / len(results)
        avg_r  = sum(r["recall"]    for r in results) / len(results)
        avg_f1 = sum(r["f1"]        for r in results) / len(results)

        print("\n" + "-" * 65)
        print(f"  Macro-average Precision : {avg_p:.3f}")
        print(f"  Macro-average Recall    : {avg_r:.3f}")
        print(f"  Macro-average F1        : {avg_f1:.3f}")
        print("=" * 65)

    return results


# ═════════════════════════════════════════════════════════════
# QUICK SELF-TEST  (run: python evaluator.py)
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from indexer import get_index

    print("=" * 55)
    print("Evaluator — Self Test")
    print("=" * 55)

    # Unit tests for metrics
    print("\n── Metric Unit Tests ────────────────────────────────")
    tests = [
        # (retrieved,         relevant,           exp_p, exp_r)
        (["a","b","c"],    ["a","b"],            2/3,   1.0),
        (["a","b"],        ["a","b","c"],        1.0,   2/3),
        ([],               ["a"],                0.0,   0.0),
        (["a"],            [],                   0.0,   0.0),
        (["x","y"],        ["a","b"],            0.0,   0.0),
    ]
    all_pass = True
    for ret, rel, exp_p, exp_r in tests:
        p = precision(ret, rel)
        r = recall(ret, rel)
        ok = abs(p - exp_p) < 1e-9 and abs(r - exp_r) < 1e-9
        status = "✓" if ok else "✗"
        print(f"  {status}  precision={p:.3f} (exp {exp_p:.3f})  "
            f"recall={r:.3f} (exp {exp_r:.3f})")
        if not ok:
            all_pass = False
    print("  All metric tests passed!" if all_pass else "  Some tests FAILED.")

    # Integration test
    print("\n── Full Evaluation (requires built index) ───────────")
    try:
        idx = get_index()
        evaluate(idx)
    except FileNotFoundError:
        print("  [SKIP] Index not found — run fetch_corpus.py and indexer.py first.")
