# main.py
# Command-line interface for the Bilingual Search Engine.
# Ties together: indexer, searcher, tfidf, spell_check, evaluator.

import sys
from indexer     import get_index
from searcher    import search
from tfidf       import rank
from spell_check import check_and_suggest
from evaluator   import evaluate
from preprocessing import detect_language


# ═════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═════════════════════════════════════════════════════════════

DIVIDER = "=" * 55
THIN    = "-" * 55


def print_banner():
    print(f"\n{DIVIDER}")
    print("        📰 Bilingual News Search Engine")
    print("        English | العربية")
    print(DIVIDER)
    print("  Commands:")
    print("    /eval   → run Precision & Recall evaluation")
    print("    /rebuild→ force rebuild the index")
    print("    /help   → show this menu")
    print("    /quit   → exit")
    print(DIVIDER)


def print_results(ranked, search_result, index):
    """
    Display ranked results with document names and scores.
    Also show spelling suggestions for any OOV terms.
    """
    lang      = search_result.get("language", "?")
    qtype     = search_result.get("query_type", "?")
    tokens    = search_result.get("query_tokens", [])
    missing   = search_result.get("missing_terms", [])

    print(f"\n  Language : {lang}")
    print(f"  Type     : {qtype}")
    print(f"  Tokens   : {tokens}")

    # ── Spelling suggestions for OOV terms ────────────────────
    if missing:
        suggestions = check_and_suggest(tokens, index)
        for term, options in suggestions.items():
            if options:
                print(f"\n  ❓ '{term}' not found.")
                print(f"     Did you mean: {' | '.join(options)}")
            else:
                print(f"\n  ❌ '{term}' not found — no suggestion available.")

    # ── Results ───────────────────────────────────────────────
    print()
    if not ranked:
        print("  No results found.")
        return

    print(f"  Found {len(ranked)} result(s):\n")
    print(f"  {'#':<4} {'Document':<25} {'Score':>8}")
    print(f"  {THIN}")

    for rank, (doc_id, score) in enumerate(ranked, 1):
        bar   = "█" * int(score * 20)   # simple visual bar
        print(f"  {rank:<4} {doc_id:<25} {score:>8.4f}  {bar}")

    print()


# ═════════════════════════════════════════════════════════════
# LANGUAGE SELECTION HELPER
# ═════════════════════════════════════════════════════════════

def resolve_language(raw_query):
    """
    Auto-detect language from the query text.
    If the result is ambiguous (nearly equal counts), ask the user.

    Returns "english" or "arabic".
    """
    lang = detect_language(raw_query)
    return lang


# ═════════════════════════════════════════════════════════════
# MAIN LOOP
# ═════════════════════════════════════════════════════════════

def main():
    # ── Load or build index ───────────────────────────────────
    print("\nLoading index...")
    try:
        index = get_index()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Run 'python fetch_corpus.py' first to download the corpus.")
        sys.exit(1)

    if index["doc_count"] == 0:
        print("[WARNING] Index contains zero documents. Results will be empty.")

    print_banner()

    # ── REPL loop ─────────────────────────────────────────────
    while True:
        try:
            raw_query = input("\nEnter query / اكتب البحث: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D or Ctrl+C — exit gracefully
            print("\n\nGoodbye! مع السلامة 👋")
            break

        # ── Empty input ───────────────────────────────────────
        if not raw_query:
            print("  ⚠️  Please enter a search term.")
            continue

        # ── Commands ──────────────────────────────────────────
        command = raw_query.lower()

        if command in ("/quit", "/exit", "quit", "exit"):
            print("\nGoodbye! مع السلامة 👋")
            break

        if command == "/help":
            print_banner()
            continue

        if command == "/eval":
            evaluate(index)
            continue

        if command == "/rebuild":
            print("\nRebuilding index from corpus...")
            try:
                index = get_index(force_rebuild=True)
                print("Index rebuilt successfully.")
            except FileNotFoundError as e:
                print(f"[ERROR] {e}")
            continue

        # ── Normal search ─────────────────────────────────────
        language = resolve_language(raw_query)

        search_result = search(raw_query, index, language=language)
        ranked        = rank(
            search_result["query_tokens"],
            search_result["matches"],
            index,
            top_n=10,
        )

        print_results(ranked, search_result, index)


# ═════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
