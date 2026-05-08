"""
searcher.py
Query processing and retrieval using the Positional Inverted Index.

Supports:
    - Boolean (AND) retrieval      →  climate change
    - Phrase queries               →  "climate change"
    - Proximity queries            →  employment /3 place
    - Wildcard queries             →  comput*

Public API
----------
    search(query, index, language=None) -> dict
"""

import re
import logging
from preprocessing import preprocess, detect_language

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════
# QUERY-TYPE DETECTION
# ═════════════════════════════════════════════════════════════

def _detect_query_type(query: str) -> str:
    """
    Classify the raw query string into a query type.

    Rules (checked in order):
        1. Wrapped in double quotes           → phrase
        2. Contains /N between two parts       → proximity
        3. Contains * or ?                     → wildcard
        4. Otherwise                           → boolean (AND)
    """
    q = query.strip()
    if q.startswith('"') and q.endswith('"') and len(q) > 2:
        return "phrase"
    if re.search(r'\S+\s+/\d+\s+\S+', q):
        return "proximity"
    if '*' in q or '?' in q:
        return "wildcard"
    return "boolean"


# ═════════════════════════════════════════════════════════════
# BOOLEAN (AND) RETRIEVAL
# ═════════════════════════════════════════════════════════════

def _boolean_search(tokens, positional_index, lang_map, language):
    """
    Intersect posting lists for all tokens (AND semantics).

    Returns only documents whose language matches *language*.
    """
    if not tokens:
        return []

    # Documents that belong to the requested language
    lang_docs = {doc_id for doc_id, lang in lang_map.items()
                 if lang == language}

    result_set = None
    for token in tokens:
        postings = positional_index.get(token, {})
        doc_ids = set(postings.keys()) & lang_docs
        if result_set is None:
            result_set = doc_ids
        else:
            result_set = result_set & doc_ids

    return sorted(result_set) if result_set else []


# ═════════════════════════════════════════════════════════════
# PHRASE SEARCH
# ═════════════════════════════════════════════════════════════

def _phrase_search(tokens, positional_index, lang_map, language):
    """
    Find documents where *tokens* appear as consecutive positions.

    Uses the positional index to verify exact adjacency.
    """
    if not tokens:
        return []
    if len(tokens) == 1:
        return _boolean_search(tokens, positional_index, lang_map, language)

    lang_docs = {doc_id for doc_id, lang in lang_map.items()
                 if lang == language}

    # Start with documents containing the first token
    first_postings = positional_index.get(tokens[0], {})
    candidates = set(first_postings.keys()) & lang_docs

    results = []
    for doc_id in candidates:
        start_positions = first_postings[doc_id]
        for start_pos in start_positions:
            match = True
            for offset, token in enumerate(tokens[1:], 1):
                token_postings = positional_index.get(token, {})
                doc_positions = token_postings.get(doc_id, [])
                if (start_pos + offset) not in doc_positions:
                    match = False
                    break
            if match:
                results.append(doc_id)
                break  # one match per document is sufficient

    return sorted(results)


# ═════════════════════════════════════════════════════════════
# PROXIMITY SEARCH
# ═════════════════════════════════════════════════════════════

def _proximity_search(tokens, distance, positional_index, lang_map, language):
    """
    Find documents where two terms appear within *distance* positions
    of each other (in either order).

    Falls back to boolean search if fewer than 2 tokens are supplied.
    """
    if len(tokens) < 2:
        return _boolean_search(tokens, positional_index, lang_map, language)

    term1, term2 = tokens[0], tokens[1]

    lang_docs = {doc_id for doc_id, lang in lang_map.items()
                 if lang == language}

    postings1 = positional_index.get(term1, {})
    postings2 = positional_index.get(term2, {})

    common_docs = (set(postings1.keys()) & set(postings2.keys())) & lang_docs

    results = []
    for doc_id in common_docs:
        pos1_list = postings1[doc_id]
        pos2_list = postings2[doc_id]
        found = False
        for p1 in pos1_list:
            for p2 in pos2_list:
                if abs(p1 - p2) <= distance:
                    found = True
                    break
            if found:
                break
        if found:
            results.append(doc_id)

    return sorted(results)


# ═════════════════════════════════════════════════════════════
# WILDCARD SEARCH
# ═════════════════════════════════════════════════════════════

def _wildcard_search(pattern, positional_index, lang_map, language):
    """
    Match index terms against a shell-style wildcard pattern.

    Supported wildcards:
        *  →  any number of characters
        ?  →  exactly one character

    Example: comput* matches compute, computer, computing, ...
    """
    lang_docs = {doc_id for doc_id, lang in lang_map.items()
                 if lang == language}

    # Convert wildcard to regex
    regex_pattern = re.escape(pattern).replace(r'\*', '.*').replace(r'\?', '.')
    regex = re.compile(f'^{regex_pattern}$')

    matching_docs = set()
    for term in positional_index:
        if regex.match(term):
            doc_ids = set(positional_index[term].keys()) & lang_docs
            matching_docs |= doc_ids

    return sorted(matching_docs)


# ═════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════

def search(query, index, language=None):
    """
    Process a user query and return matching documents.

    Parameters
    ----------
    query    : raw query string from the user.
    index    : full index dict from indexer.get_or_build_index().
               Must contain: positional_index, tf_table, idf_table,
               doc_norms, lang_map.
    language : "english" or "arabic".  Auto-detected from the query
               text when None.

    Returns
    -------
    dict with keys:
        language      : str   – detected / forced language
        query_type    : str   – "boolean" | "phrase" | "proximity" | "wildcard"
        query_tokens  : list  – preprocessed tokens sent to the index
        matches       : list  – doc_ids of matching documents
        missing_terms : list  – tokens not found in the index (OOV)
    """
    # ── empty / blank guard ────────────────────────────────────────────────
    if not query or not query.strip():
        return {
            "language":      language or "unknown",
            "query_type":    "boolean",
            "query_tokens":  [],
            "matches":       [],
            "missing_terms": [],
        }

    # ── auto-detect language ───────────────────────────────────────────────
    if language is None:
        language = detect_language(query)

    positional_index = index.get("positional_index", {})
    lang_map         = index.get("lang_map", {})

    query_type = _detect_query_type(query)

    # ── dispatch by query type ─────────────────────────────────────────────
    if query_type == "phrase":
        raw_text = query.strip('"')
        tokens   = preprocess(raw_text, language)
        matches  = _phrase_search(tokens, positional_index, lang_map, language)

    elif query_type == "proximity":
        match = re.match(r'(.+?)\s*/(\d+)\s*(.+)', query)
        if match:
            term1_text = match.group(1).strip()
            distance   = int(match.group(2))
            term2_text = match.group(3).strip()
            t1 = preprocess(term1_text, language)
            t2 = preprocess(term2_text, language)
            tokens  = t1 + t2
            matches = _proximity_search(
                tokens, distance, positional_index, lang_map, language
            )
        else:
            # Regex didn't match — fall back to boolean
            tokens  = preprocess(query, language)
            matches = _boolean_search(
                tokens, positional_index, lang_map, language
            )

    elif query_type == "wildcard":
        # Preprocess only the non-wildcard characters.
        # We temporarily replace wildcards with Unicode placeholders so the
        # preprocessor doesn't strip them, then restore them afterwards.
        placeholder_star = '\u2732'  # ✲
        placeholder_qm   = '\u2738'  # ✸
        safe_query = query.replace('*', placeholder_star).replace('?', placeholder_qm)
        processed  = preprocess(safe_query, language)

        if processed:
            pattern = processed[0].replace(placeholder_star, '*').replace(placeholder_qm, '?')
        else:
            pattern = query.strip()

        tokens  = [pattern]
        matches = _wildcard_search(pattern, positional_index, lang_map, language)

    else:  # boolean (AND)
        tokens  = preprocess(query, language)
        matches = _boolean_search(tokens, positional_index, lang_map, language)

    # ── identify OOV (missing) terms ───────────────────────────────────────
    missing_terms = [
        t for t in tokens
        if t not in positional_index and '*' not in t and '?' not in t
    ]

    return {
        "language":      language,
        "query_type":    query_type,
        "query_tokens":  tokens,
        "matches":       matches,
        "missing_terms": missing_terms,
    }


# ═════════════════════════════════════════════════════════════
# CLI SELF-TEST  (run: python searcher.py)
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from indexer import get_or_build_index

    print("=" * 55)
    print("Searcher — Self Test")
    print("=" * 55)

    try:
        idx = get_or_build_index()
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        raise SystemExit(1)

    test_queries = [
        ("climate change",           "english"),
        ('"artificial intelligence"', "english"),
        ("employment /3 place",       "english"),
        ("comput*",                   "english"),
        ("تغير المناخ",              "arabic"),
    ]

    for q, lang in test_queries:
        result = search(q, idx, language=lang)
        print(f"\n  Query   : {q}")
        print(f"  Type    : {result['query_type']}")
        print(f"  Tokens  : {result['query_tokens']}")
        print(f"  Matches : {result['matches']}")
        if result["missing_terms"]:
            print(f"  OOV     : {result['missing_terms']}")