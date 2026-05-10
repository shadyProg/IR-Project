# main.py
# Command-line interface for the Bilingual Search Engine.
# Ties together: indexer, searcher, tfidf, spell_check, evaluator.

import os
import sys
import re

# Fix Windows console encoding for Arabic output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

#  Make src/ importable from the project root 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from indexer       import get_or_build_index
from searcher      import search
from tfidf         import rank
from spell_check   import get_suggestions
from evaluator     import evaluate
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
        for term in missing:
            suggestions = get_suggestions(term, index)
            if suggestions:
                print(f"\n  ❓ '{term}' not found.")
                print(f"     Did you mean: {' | '.join(suggestions)}")
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

    for pos, (doc_id, score) in enumerate(ranked, 1):
        bar   = "█" * int(score * 20)   # simple visual bar
        print(f"  {pos:<4} {doc_id:<25} {score:>8.4f}  {bar}")

    print()



def resolve_language(raw_query):
    """
    Auto-detect language from the query text.
    If the result is ambiguous (nearly equal counts), ask the user.

    Returns "english" or "arabic".
    """
    lang = detect_language(raw_query)
    return lang


def main():

    print("\nLoading index...")
    try:
        index = get_or_build_index()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Make sure the corpus/ directory exists with english/ and arabic/ sub-folders.")
        sys.exit(1)

    if not index.get("lang_map"):
        print("[WARNING] Index contains zero documents. Results will be empty.")

    print_banner()

    
    while True:
        try:
            raw_query = input("\nEnter query / اكتب البحث: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D or Ctrl+C — exit gracefully
            print("\n\nGoodbye! مع السلامة 👋")
            break

        
        if not raw_query:
            print("  ⚠️  Please enter a search term.")
            continue


        pattern = r'^[/a-zA-Z\u0600-\u06FF\s]+$'

        if not re.fullmatch(pattern, raw_query):
            print("\nenter only english or arabic characters, please. ")
            continue

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
                index = get_or_build_index(force_rebuild=True)
                print("Index rebuilt successfully.")
            except FileNotFoundError as e:
                print(f"[ERROR] {e}")
            continue

    
        language = resolve_language(raw_query)

        search_result = search(raw_query, index, language=language)
        ranked        = rank(
            search_result["query_tokens"],
            search_result["matches"],
            index,
            top_n=10,
        )

        print_results(ranked, search_result, index)


if __name__ == "__main__":
    main()
