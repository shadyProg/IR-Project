"""
tfidf.py
Ranks candidate documents by TF-IDF cosine similarity to the query.

The indexer already computed and stored:
    tf_table  : { doc_id -> { term -> tf_weight } }   tf = 1 + log10(count)
    idf_table : { term -> idf_weight }                idf = log10(N / df)
    doc_norms : { doc_id -> L2 norm }                 pre-computed for speed

This file only needs to:
    1. Build the query vector  (same TF-IDF formula as the indexer)
    2. Compute cosine similarity for each candidate document
    3. Return results sorted highest score first

Public API
----------
    rank(query_tokens, candidate_doc_ids, index, top_n=10) -> list[tuple]
        Main entry point. Returns up to top_n (doc_id, score) pairs,
        sorted from most to least relevant.

Formulas
--------
    TF weight       = 1 + log10(count)
    IDF weight      = log10(N / df)          [pre-computed by indexer]
    TF-IDF weight   = TF x IDF
    Cosine sim      = dot(q, d) / (|q| x |d|)
"""

import math
import logging

logger = logging.getLogger(__name__)



def _build_query_vector(query_tokens: list, idf_table: dict) -> dict:
    """
    Build a TF-IDF weighted vector for the query.

    Uses the same 1 + log10(count) formula as the indexer so that
    query weights and document weights are on the same scale.

    A term that does not exist in the index gets IDF = 0.0, so its
    weight is 0 and it contributes nothing to the score (OOV behaviour).

    Returns
    -------
    dict { term -> tfidf_weight }  (only terms with weight > 0 are included)
    """
    # count how many times each term appears in the query
    term_counts: dict[str, int] = {}
    for token in query_tokens:
        term_counts[token] = term_counts.get(token, 0) + 1

    query_vector: dict[str, float] = {}
    for term, count in term_counts.items():
        tf     = 1.0 + math.log10(count)    # dampened term frequency
        idf    = idf_table.get(term, 0.0)   # 0.0 if term not in index
        weight = tf * idf
        if weight > 0:
            query_vector[term] = weight

    return query_vector




def _cosine_similarity(
    query_vector:   dict,
    doc_id:         str,
    tf_table:       dict,
    idf_table:      dict,
    doc_norms:      dict,
) -> float:
    """
    Compute cosine similarity between the query vector and one document.

    cos(q, d) = dot(q, d) / (|q| x |d|)

    dot(q, d) sums  q_weight(t) x d_weight(t)  for every term t that
    appears in BOTH the query and the document. Terms in only one of
    the two contribute zero and are skipped.

    Returns 0.0 when either norm is zero (no division by zero).
    """
    doc_tf_weights = tf_table.get(doc_id, {})
    doc_norm       = doc_norms.get(doc_id, 0.0)

    # dot product over shared terms
    dot_product = 0.0
    for term, q_weight in query_vector.items():
        doc_tf  = doc_tf_weights.get(term, 0.0)
        doc_idf = idf_table.get(term, 0.0)
        dot_product += q_weight * (doc_tf * doc_idf)

    # query L2 norm
    query_norm = math.sqrt(sum(w ** 2 for w in query_vector.values()))

    denominator = query_norm * doc_norm
    return dot_product / denominator if denominator > 0 else 0.0



def rank(
    query_tokens:      list,
    candidate_doc_ids: list,
    index:             dict,
    top_n:             int = 10,
) -> list:
    """
    Rank candidate documents by cosine similarity to the query.

    Parameters
    ----------
    query_tokens      : preprocessed query terms e.g. ['climat', 'chang']
    candidate_doc_ids : doc_ids returned by the searcher
    index             : full index dict from indexer.py (needs tf_table,
                        idf_table, doc_norms)
    top_n             : how many results to return  (0 = return all)

    Returns
    -------
    List of (doc_id, score) tuples sorted from highest to lowest score.

    Edge cases handled
    ------------------
    - Empty query or candidates        -> []
    - All query terms OOV              -> all scores 0.0
    - Zero document norm               -> that doc scores 0.0 (ranked last)
    - top_n <= 0                       -> all results returned (no slice)
    """


    if not query_tokens or not candidate_doc_ids:
        return []

    tf_table  = index.get("tf_table",  {})
    idf_table = index.get("idf_table", {})
    doc_norms = index.get("doc_norms", {})


    query_vector = _build_query_vector(query_tokens, idf_table)

    if not query_vector:
        # every query term was OOV — cannot rank meaningfully
        logger.warning("Query vector is empty (all terms OOV). Returning unranked.")
        return [(doc_id, 0.0) for doc_id in candidate_doc_ids]

   
    scored = []
    for doc_id in candidate_doc_ids:
        score = _cosine_similarity(query_vector, doc_id, tf_table, idf_table, doc_norms)
        scored.append((doc_id, score))

    #  sort highest first 
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return scored if top_n <= 0 else scored[:top_n]


#  CLI demo 

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: python tfidf.py <index.json> <query terms...>")
        print("Example: python tfidf.py index.json climate change")
        sys.exit(1)

    index_path  = sys.argv[1]
    query_terms = sys.argv[2:]   # remaining args are the raw query words

    try:
        with open(index_path, encoding="utf-8") as fh:
            idx = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR loading index: {e}")
        sys.exit(1)

    # query terms should already be preprocessed when called from main.py;
    # here we use them as-is for quick testing
    results = rank(query_terms, list(idx["lang_map"].keys()), idx)

    if not results:
        print("No results.")
    else:
        print(f"\nTop results for: {' '.join(query_terms)}\n")
        for i, (doc_id, score) in enumerate(results, 1):
            bar = "█" * int(score * 20)
            print(f"  {i}. {doc_id:<25} Score: {score:.4f}  {bar}")