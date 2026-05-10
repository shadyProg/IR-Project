"""
spell_check.py
Two-stage spelling correction:
    Stage 1 — Jaccard Similarity on character bigrams (fast broad filter)
    Stage 2 — Levenshtein (edit) distance on surviving candidates (precise ranking)

Public API
----------
    get_suggestions(term, index, top_n=3) -> list[str]
        Main entry point. Returns up to top_n correction suggestions,
        or an empty list when no correction is needed / possible.

    get_kgrams(word, k=2) -> set[str]
        Returns the set of k-length character n-grams for a word.

    jaccard_similarity(set_a, set_b) -> float
        |A ∩ B| / |A ∪ B|. Returns 0.0 for two empty sets.

    levenshtein_distance(s1, s2) -> int
        Rolling-row DP in O(m × n) time, O(n) space.
"""

import logging
from typing import Iterable

logger = logging.getLogger(__name__)


JACCARD_THRESHOLD: float = 0.2   # Stage 1 filter; candidates below this are dropped
DEFAULT_K:         int   = 2     # bigram size used for Jaccard


# ── n-gram generation ──────────────────────────────────────────────────────────

def get_kgrams(word: str, k: int = DEFAULT_K) -> set[str]:
    """
    Return the set of k-length character substrings (k-grams) of *word*.

    Edge cases
    ----------
    - k ≤ 0       : warns and defaults k to 2.
    - Empty word   : returns an empty set.
    - len(word) < k: returns a set containing *word* itself (single gram).
    """
    if k <= 0:
        logger.warning("k must be > 0; defaulting to 2.")
        k = DEFAULT_K

    if not word:
        return set()

    if len(word) < k:
        return {word}

    return {word[i: i + k] for i in range(len(word) - k + 1)}


# ── Jaccard similarity ─────────────────────────────────────────────────────────

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """
    Compute Jaccard similarity: |A ∩ B| / |A ∪ B|.

    Returns 0.0 when both sets are empty (avoids division by zero).
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union        = len(set_a | set_b)
    return intersection / union


# ── Levenshtein (edit) distance ────────────────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Minimum number of single-character edits (insert / delete / substitute)
    to transform s1 into s2.

    Implemented with a rolling single-row DP:
        Time  : O(m × n)
        Space : O(n)

    Edge cases
    ----------
    - Both strings empty : returns 0.
    - One string empty   : returns length of the other.
    """
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    m, n = len(s1), len(s2)

    # prev[j] = edit distance between s1[:i] and s2[:j]
    prev = list(range(n + 1))

    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]               # characters match — free
            else:
                curr[j] = 1 + min(
                    prev[j],       # delete from s1
                    curr[j - 1],   # insert into s1
                    prev[j - 1],   # substitute
                )
        prev = curr

    return prev[n]


# ── main correction function ───────────────────────────────────────────────────

def get_suggestions(
    term:    str,
    index:   dict,
    top_n:   int = 3,
    k:       int = DEFAULT_K,
    threshold: float = JACCARD_THRESHOLD,
) -> list[str]:
    """
    Return up to *top_n* spelling suggestions for *term*.

    Parameters
    ----------
    term      : the (possibly misspelled) query term, already preprocessed.
    index     : the full index dict returned by indexer.build_index().
                Must contain key "positional_index".
    top_n     : maximum number of suggestions to return.
    k         : k-gram size for Jaccard filter (default 2 = bigrams).
    threshold : minimum Jaccard score to pass Stage 1 (default 0.2).

    Returns
    -------
    List of suggested correction strings, best first.
    Returns [] when:
        - term is None or empty
        - term IS already in the index (no correction needed)
        - the index is empty
        - no candidates pass the Jaccard threshold

    Algorithm
    ---------
    Stage 1 — Jaccard filter
        Build bigrams of *term*. For every known index term compute Jaccard
        score. Keep those ≥ threshold.

    Stage 2 — Levenshtein ranking
        Compute edit distance for surviving candidates.
        Sort ascending by distance (ties broken alphabetically).
        Return top_n results.
    """
    
    if not term:
        return []

    vocab: Iterable[str] = index.get("positional_index", {}).keys()


    vocab_set = set(vocab)
    if not vocab_set:
        return []

    
    if term in vocab_set:
        return []


    term_grams = get_kgrams(term, k=k)

    candidates: list[str] = []
    for known_term in vocab_set:
        known_grams = get_kgrams(known_term, k=k)
        score = jaccard_similarity(term_grams, known_grams)
        if score >= threshold:
            candidates.append(known_term)

    if not candidates:
        # No candidates survived the filter
        return []


    ranked = sorted(
        candidates,
        key=lambda candidate: (levenshtein_distance(term, candidate), candidate),
    )

    return ranked[:top_n] if top_n > 0 else ranked



if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 3:
        print("Usage: python spell_check.py <index.json> <misspelled_term>")
        sys.exit(1)

    index_path, query_term = sys.argv[1], sys.argv[2]

    try:
        with open(index_path, encoding="utf-8") as fh:
            idx = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR loading index: {e}")
        sys.exit(1)

    suggestions = get_suggestions(query_term, idx)
    if suggestions:
        print(f"Did you mean: {' | '.join(suggestions)}")
    else:
        print("No suggestion available.")   