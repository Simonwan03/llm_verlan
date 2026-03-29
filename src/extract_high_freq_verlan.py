"""
extract_high_freq_verlan.py
============================
Count verlan occurrences in an OpenSubtitles-style French text corpus,
aggregate variants by canonical form, and export frequency tables.

Usage
-----
python src/extract_high_freq_verlan.py \
    --verlan_csv  data/processed/verlan_database.csv \
    --corpus      data/raw/opensubtitles_v1_clean.txt \
    --output_dir  results \
    --min_count   5 \
    --expand_plural

All arguments have sensible defaults; the only truly required one is --corpus.
"""

import re
import sys
import argparse
import unicodedata
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ---------------------------------------------------------------------------
# Optional progress bar (graceful fallback if tqdm is not installed)
# ---------------------------------------------------------------------------
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


# ===========================================================================
# 1.  Verlan table — loading and cleaning
# ===========================================================================

APOSTROPHE_RE = re.compile(r"['\u2019\u02bc]")  # right single quote → straight

REQUIRED_COLS = {"verlan_form", "base_form"}


def normalise_apostrophe(text: str) -> str:
    """Replace all typographic apostrophe variants with a plain ASCII apostrophe."""
    return APOSTROPHE_RE.sub("'", text)


def make_canonical(form: str) -> str:
    """
    Produce a canonical key for grouping spelling variants.

    Steps:
      1. lower-case
      2. strip leading/trailing whitespace
      3. unify apostrophes → '
      4. remove spaces and hyphens (so 'balle-peau' and 'ballepeau' share a key)

    We do NOT strip accents here: 'câblé' and 'cable' are different words.
    """
    s = form.strip().lower()
    s = normalise_apostrophe(s)
    s = re.sub(r"[\s\-]+", "", s)
    return s


def load_verlan_table(csv_path: Path) -> pd.DataFrame:
    """
    Read verlan_database.csv, clean it, and add helper columns.

    Returns a deduplicated DataFrame with at least:
      verlan_form, base_form, pos, is_multiword, has_hyphen,
      token_count, canonical_form
    """
    if not csv_path.exists():
        sys.exit(f"[ERROR] Verlan CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # --- check required columns ---
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns in CSV: {missing}")

    # --- keep all columns but work on core ones ---
    df["verlan_form"] = (
        df["verlan_form"]
        .astype(str)
        .str.strip()
        .str.lower()
        .apply(normalise_apostrophe)
    )
    df["base_form"] = (
        df["base_form"]
        .astype(str)
        .str.strip()
        .str.lower()
        .apply(normalise_apostrophe)
    )

    # drop rows where either core field ended up empty after cleaning
    df = df[df["verlan_form"].str.len() > 0]
    df = df[df["base_form"].str.len() > 0]

    # --- helper columns ---
    df["is_multiword"] = df["verlan_form"].str.contains(r" ", regex=False)
    df["has_hyphen"] = df["verlan_form"].str.contains(r"-", regex=False)
    df["token_count"] = df["verlan_form"].str.split().apply(len)
    df["canonical_form"] = df["verlan_form"].apply(make_canonical)

    # --- deduplicate by (verlan_form, base_form) pairs ---
    df = df.drop_duplicates(subset=["verlan_form", "base_form"]).reset_index(drop=True)

    print(f"[INFO] Verlan table loaded: {len(df)} entries, "
          f"{df['canonical_form'].nunique()} canonical forms.")
    return df


# ===========================================================================
# 2.  Building match patterns
# ===========================================================================

# Tokenisation pattern:  word chars OR apostrophe OR hyphen, anchored by \b
# We use a simple but robust approach: tokenise on whitespace + punctuation,
# keeping contractions like "j'ai" as one token.
TOKENISE_RE = re.compile(r"[a-zA-ZÀ-ÿ\u0300-\u036f\u00c0-\u024f''`\-]+")


def build_token_pattern(form: str, expand_plural: bool) -> re.Pattern:
    """
    Build a compiled regex that matches the verlan *word* (not phrase) as a
    standalone token.  Optional plural expansion appends an optional 's'.
    """
    escaped = re.escape(form)
    if expand_plural:
        escaped = escaped + r"s?"
    # word-boundary anchors — safe for French because our tokens are alphabetic
    return re.compile(r"(?<![a-zA-ZÀ-ÿ\u00c0-\u024f])" + escaped
                      + r"(?![a-zA-ZÀ-ÿ\u00c0-\u024f])", re.IGNORECASE)


def build_phrase_pattern(form: str, expand_plural: bool) -> re.Pattern:
    """
    Build a compiled regex for a multi-word / hyphenated phrase.
    Internal spaces are replaced by \\s+ to tolerate minor whitespace variation.
    Hyphens are made optional (match '-' or nothing) to merge orthographic
    variants such as 'balle-peau' and 'balle peau'.
    Plural expansion only on the last token.
    """
    # normalise the form for phrase matching
    # split on space or hyphen
    parts = re.split(r"[\s\-]+", form)
    parts_escaped = [re.escape(p) for p in parts if p]

    if expand_plural and parts_escaped:
        parts_escaped[-1] = parts_escaped[-1] + r"s?"

    # Allow any combo of space / hyphen between parts
    connector = r"[\s\-]+"
    inner = connector.join(parts_escaped)

    return re.compile(
        r"(?<![a-zA-ZÀ-ÿ\u00c0-\u024f])" + inner
        + r"(?![a-zA-ZÀ-ÿ\u00c0-\u024f])",
        re.IGNORECASE,
    )


def build_patterns(df: pd.DataFrame, expand_plural: bool) -> dict:
    """
    Return a dict:  canonical_form -> {
        "pattern": compiled regex,
        "is_multiword": bool,
        "has_hyphen": bool,
        "variants": [verlan_form, ...],
        "base_forms": [base_form, ...],
    }

    Patterns for a canonical group are built from ALL variants merged via
    alternation, which avoids double-counting within the group.
    """
    groups: dict = {}

    for canon, sub in df.groupby("canonical_form"):
        variants = sub["verlan_form"].tolist()
        base_forms = sub["base_form"].unique().tolist()
        is_multiword = bool(sub["is_multiword"].any())
        has_hyphen = bool(sub["has_hyphen"].any())

        # Build one alternation pattern per canonical group.
        # Sort longest first to prefer maximal match.
        sorted_variants = sorted(variants, key=len, reverse=True)
        escaped_alts = []
        for v in sorted_variants:
            if expand_plural:
                escaped_alts.append(re.escape(v) + r"s?")
            else:
                escaped_alts.append(re.escape(v))

        # Also add hyphen/space flexible connectors for phrase variants
        # by replacing internal spaces/hyphens with [\s\-]+ in each alt
        flex_alts = []
        for alt in escaped_alts:
            # re.escape turns spaces into '\ ' and hyphens into '\-'
            # Undo and replace with flexible connector
            flex = re.sub(r"(\\\ |\\-)", r"[\\s\\-]+", alt)
            flex_alts.append(flex)

        # Combine: original escaped alts OR flexible alts
        all_alts = list(dict.fromkeys(escaped_alts + flex_alts))  # dedup, order-preserving
        pattern_body = "|".join(f"(?:{a})" for a in all_alts)

        try:
            pat = re.compile(
                r"(?<![a-zA-ZÀ-ÿ\u00c0-\u024f])(?:"
                + pattern_body
                + r")(?![a-zA-ZÀ-ÿ\u00c0-\u024f])",
                re.IGNORECASE,
            )
        except re.error as exc:
            print(f"[WARN] Could not compile pattern for '{canon}': {exc} — skipping.")
            continue

        groups[canon] = {
            "pattern": pat,
            "is_multiword": is_multiword,
            "has_hyphen": has_hyphen,
            "variants": variants,
            "base_forms": base_forms,
            "variant_count": len(variants),
        }

    return groups


# ===========================================================================
# 3.  Corpus streaming & counting
# ===========================================================================

def tokenise_line(line: str) -> list[str]:
    """
    Very lightweight French tokeniser: extract word-like tokens (including
    apostrophes inside contractions and hyphenated compounds).
    """
    return TOKENISE_RE.findall(line)


def count_corpus(
    corpus_path: Path,
    patterns: dict,
    encoding: str = "utf-8",
) -> tuple[dict, int, int]:
    """
    Stream through corpus_path line by line.

    Returns
    -------
    counts : dict  canonical_form -> {"raw_count": int, "doc_count": int}
    total_lines : int
    total_tokens : int
    """
    if not corpus_path.exists():
        sys.exit(f"[ERROR] Corpus file not found: {corpus_path}")

    # Initialise counters
    counts: dict = {canon: {"raw_count": 0, "doc_count": 0} for canon in patterns}

    total_lines = 0
    total_tokens = 0

    with corpus_path.open(encoding=encoding, errors="replace") as fh:
        for line in tqdm(fh, desc="Scanning corpus", unit=" lines"):
            line_lower = line.rstrip("\n").lower()
            if not line_lower.strip():
                continue

            total_lines += 1
            total_tokens += len(tokenise_line(line_lower))

            for canon, info in patterns.items():
                matches = list(info["pattern"].finditer(line_lower))
                if matches:
                    counts[canon]["raw_count"] += len(matches)
                    counts[canon]["doc_count"] += 1

    return counts, total_lines, total_tokens


# ===========================================================================
# 4.  Assembling the result table
# ===========================================================================

def build_result_df(
    patterns: dict,
    counts: dict,
    total_tokens: int,
) -> pd.DataFrame:
    """
    Merge pattern metadata with raw counts into a single DataFrame.
    """
    rows = []
    total_verlan_hits = sum(c["raw_count"] for c in counts.values())

    for canon, info in patterns.items():
        rc = counts[canon]["raw_count"]
        dc = counts[canon]["doc_count"]

        pmw = (rc / total_tokens * 1_000_000) if total_tokens > 0 else 0.0
        share = (rc / total_verlan_hits) if total_verlan_hits > 0 else 0.0

        rows.append({
            "canonical_verlan": canon,
            "variants": "|".join(info["variants"]),
            "base_forms": "|".join(info["base_forms"]),
            "is_multiword": info["is_multiword"],
            "has_hyphen": info["has_hyphen"],
            "variant_count": info["variant_count"],
            "raw_count": rc,
            "doc_count": dc,
            "pmw": round(pmw, 4),
            "share_of_all_verlan_hits": round(share, 6),
        })

    df = pd.DataFrame(rows).sort_values("raw_count", ascending=False).reset_index(drop=True)
    return df, total_verlan_hits


# ===========================================================================
# 5.  Output & summary
# ===========================================================================

def save_outputs(
    df: pd.DataFrame,
    output_dir: Path,
    min_count: int,
) -> pd.DataFrame:
    """Write full and high-frequency CSVs; return the high-freq subset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / "verlan_frequency_full.csv"
    df.to_csv(full_path, index=False, encoding="utf-8-sig")
    print(f"[OUTPUT] Full table ({len(df)} rows) → {full_path}")

    highfreq = df[df["raw_count"] >= min_count].copy()
    hf_path = output_dir / "verlan_frequency_highfreq.csv"
    highfreq.to_csv(hf_path, index=False, encoding="utf-8-sig")
    print(f"[OUTPUT] High-freq table ({len(highfreq)} rows, raw_count >= {min_count}) → {hf_path}")

    return highfreq


def print_summary(
    df: pd.DataFrame,
    highfreq: pd.DataFrame,
    total_lines: int,
    total_tokens: int,
    total_verlan_hits: int,
    min_count: int,
    top_n: int = 20,
) -> None:
    """Print a human-readable summary to stdout."""
    sep = "=" * 65
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    print(f"  Total corpus lines     : {total_lines:>12,}")
    print(f"  Total tokens           : {total_tokens:>12,}")
    print(f"  Total verlan hits      : {total_verlan_hits:>12,}")
    print(f"  Canonical verlan items : {len(df):>12,}")
    print(f"  High-freq items        : {len(highfreq):>12,}  (raw_count >= {min_count})")
    print(sep)

    top = df.head(top_n)
    if top.empty:
        print("  No matches found.")
    else:
        print(f"\n  Top {min(top_n, len(top))} verlan by raw count:")
        print(f"  {'canonical':25s}  {'raw':>7}  {'pmw':>9}  {'variants'}")
        print("  " + "-" * 61)
        for _, row in top.iterrows():
            variants_preview = row["variants"][:40] + ("…" if len(row["variants"]) > 40 else "")
            print(f"  {row['canonical_verlan']:25s}  {row['raw_count']:>7,}"
                  f"  {row['pmw']:>9.2f}  {variants_preview}")
    print(sep + "\n")


# ===========================================================================
# 6.  CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count verlan frequency in a French subtitle corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--verlan_csv",
        type=Path,
        default=Path("data/processed/verlan_database.csv"),
        help="Path to verlan_database.csv",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to plain-text corpus file (one sentence/line).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Directory to write output CSV files.",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=20,
        help="Minimum raw_count for inclusion in high-freq output.",
    )
    parser.add_argument(
        "--expand_plural",
        action="store_true",
        default=False,
        help="Also count '<verlan_form>s' as a match (basic plural expansion).",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Encoding of the corpus file.",
    )
    return parser.parse_args()


# ===========================================================================
# 7.  Main
# ===========================================================================

def main() -> None:
    args = parse_args()

    print(f"[INFO] Verlan CSV  : {args.verlan_csv}")
    print(f"[INFO] Corpus      : {args.corpus}")
    print(f"[INFO] Output dir  : {args.output_dir}")
    print(f"[INFO] Min count   : {args.min_count}")
    print(f"[INFO] Expand plural: {args.expand_plural}")

    # --- Step 1: load and clean verlan table ---
    df_verlan = load_verlan_table(args.verlan_csv)

    # --- Step 2: build compiled regex patterns per canonical group ---
    print("[INFO] Compiling patterns …")
    patterns = build_patterns(df_verlan, expand_plural=args.expand_plural)
    print(f"[INFO] {len(patterns)} canonical patterns compiled.")

    # --- Step 3: stream corpus and count ---
    counts, total_lines, total_tokens = count_corpus(
        args.corpus, patterns, encoding=args.encoding
    )

    # --- Step 4: assemble result dataframe ---
    result_df, total_verlan_hits = build_result_df(patterns, counts, total_tokens)

    # --- Step 5: save outputs ---
    highfreq_df = save_outputs(result_df, args.output_dir, args.min_count)

    # --- Step 6: print summary ---
    print_summary(
        result_df,
        highfreq_df,
        total_lines,
        total_tokens,
        total_verlan_hits,
        args.min_count,
    )


if __name__ == "__main__":
    main()
