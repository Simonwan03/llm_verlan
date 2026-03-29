from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.attention_analysis import compute_attention_breakdown, compute_incoming_attention
from src.hidden_analysis import (
    cosine_similarity_list,
    forward_pass,
    layerwise_sentence_representations,
    layerwise_word_representations,
)
from src.load_model import load_model
from src.token_utils import find_target_span

DEFAULT_PAIRS_CSV = "data/processed/verlan_probe_pairs.csv"
DEFAULT_TEMPLATES_CSV = "data/processed/verlan_probe_templates.csv"
DEFAULT_OUTPUT_DIR = "outputs/verlan_probe"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a minimal verlan vs. standard probing experiment with a Hugging Face encoder."
    )
    parser.add_argument("--model-name", default="camembert-base", help="Model checkpoint name.")
    parser.add_argument("--pairs-csv", default=DEFAULT_PAIRS_CSV, help="CSV with verlan/standard pairs.")
    parser.add_argument(
        "--templates-csv",
        default=DEFAULT_TEMPLATES_CSV,
        help="CSV with template groups and sentence templates.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV outputs and metadata will be written.",
    )
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cpu or cuda.")
    parser.add_argument("--limit-pairs", type=int, default=None, help="Optional cap for quick smoke tests.")
    return parser.parse_args()


def load_pairs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rename_map = {}
    if "verlan_form" in df.columns and "verlan" not in df.columns:
        rename_map["verlan_form"] = "verlan"
    if "base_form" in df.columns and "standard" not in df.columns:
        rename_map["base_form"] = "standard"
    df = df.rename(columns=rename_map)

    required = {"verlan", "standard"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Pairs CSV is missing columns: {sorted(missing)}")

    df["verlan"] = df["verlan"].astype(str).str.strip()
    df["standard"] = df["standard"].astype(str).str.strip()
    df = df[(df["verlan"] != "") & (df["standard"] != "")].copy()
    if "template_group" not in df.columns:
        df["template_group"] = "default"
    return df.reset_index(drop=True)


def load_templates(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "template" not in df.columns:
        raise ValueError("Templates CSV must include a `template` column.")
    if "template_group" not in df.columns:
        df["template_group"] = "default"
    df["template"] = df["template"].astype(str).str.strip()
    df = df[df["template"] != ""].copy()
    return df.reset_index(drop=True)


def build_sentence_jobs(pairs_df: pd.DataFrame, templates_df: pd.DataFrame) -> pd.DataFrame:
    merged = pairs_df.merge(templates_df, how="left", on="template_group", validate="many_to_many")
    missing_templates = merged["template"].isna()
    if missing_templates.any():
        missing_groups = sorted(merged.loc[missing_templates, "template_group"].unique().tolist())
        raise ValueError(f"No templates found for groups: {missing_groups}")
    merged["sentence_verlan"] = merged.apply(lambda row: row["template"].replace("X", row["verlan"]), axis=1)
    merged["sentence_standard"] = merged.apply(
        lambda row: row["template"].replace("X", row["standard"]),
        axis=1,
    )
    return merged


def json_cell(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def metric_at(values, idx: int):
    """Safely fetch a layer metric, returning None when attention is unavailable."""
    if values is None:
        return None
    if idx < 0 or idx >= len(values):
        return None
    return values[idx]


def collect_tokenization_row(
    pair_row,
    variant: str,
    sentence: str,
    tokens,
    span,
    target_tokens,
    char_span,
):
    return {
        "verlan": pair_row["verlan"],
        "standard": pair_row["standard"],
        "template_group": pair_row["template_group"],
        "template": pair_row["template"],
        "variant": variant,
        "target_word": pair_row["verlan"] if variant == "verlan" else pair_row["standard"],
        "sentence": sentence,
        "sentence_tokens": json_cell(tokens),
        "target_tokens": json_cell(target_tokens),
        "token_span_start": span[0],
        "token_span_end": span[1],
        "char_span_start": char_span[0],
        "char_span_end": char_span[1],
        "num_sentence_tokens": len(tokens),
        "num_target_subwords": len(target_tokens),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_df = load_pairs(args.pairs_csv)
    templates_df = load_templates(args.templates_csv)

    if args.limit_pairs is not None:
        pairs_df = pairs_df.head(args.limit_pairs).copy()

    tokenizer, model, device = load_model(args.model_name, args.device)
    jobs_df = build_sentence_jobs(pairs_df, templates_df)

    tokenization_rows = []
    layer_rows = []
    failures = []

    for _, pair_row in jobs_df.iterrows():
        sent_v = pair_row["sentence_verlan"]
        sent_s = pair_row["sentence_standard"]

        try:
            tokens_v, enc_v, span_v, word_tokens_v, char_span_v = find_target_span(
                tokenizer,
                sent_v,
                pair_row["verlan"],
            )
            tokens_s, enc_s, span_s, word_tokens_s, char_span_s = find_target_span(
                tokenizer,
                sent_s,
                pair_row["standard"],
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            failures.append(
                {
                    "verlan": pair_row["verlan"],
                    "standard": pair_row["standard"],
                    "template": pair_row["template"],
                    "error": str(exc),
                }
            )
            continue

        tokenization_rows.append(
            collect_tokenization_row(pair_row, "verlan", sent_v, tokens_v, span_v, word_tokens_v, char_span_v)
        )
        tokenization_rows.append(
            collect_tokenization_row(pair_row, "standard", sent_s, tokens_s, span_s, word_tokens_s, char_span_s)
        )

        hidden_v, attn_v = forward_pass(model, enc_v, device)
        hidden_s, attn_s = forward_pass(model, enc_s, device)

        word_reps_v = layerwise_word_representations(hidden_v, span_v)
        word_reps_s = layerwise_word_representations(hidden_s, span_s)
        word_sims = cosine_similarity_list(word_reps_v, word_reps_s)

        sent_reps_v = layerwise_sentence_representations(
            hidden_v,
            enc_v["attention_mask"].to(device),
            enc_v["special_tokens_mask"].to(device),
        )
        sent_reps_s = layerwise_sentence_representations(
            hidden_s,
            enc_s["attention_mask"].to(device),
            enc_s["special_tokens_mask"].to(device),
        )
        sent_sims = cosine_similarity_list(sent_reps_v, sent_reps_s)

        has_attentions = bool(attn_v) and bool(attn_s)
        attn_breakdown_v = compute_attention_breakdown(attn_v, span_v) if has_attentions else None
        attn_breakdown_s = compute_attention_breakdown(attn_s, span_s) if has_attentions else None
        incoming_v = compute_incoming_attention(attn_v, span_v) if has_attentions else None
        incoming_s = compute_incoming_attention(attn_s, span_s) if has_attentions else None

        for layer_idx, (word_sim, sent_sim) in enumerate(zip(word_sims, sent_sims)):
            is_embedding = layer_idx == 0
            attn_idx = layer_idx - 1
            layer_rows.append(
                {
                    "model_name": args.model_name,
                    "verlan": pair_row["verlan"],
                    "standard": pair_row["standard"],
                    "template_group": pair_row["template_group"],
                    "template": pair_row["template"],
                    "sentence_verlan": sent_v,
                    "sentence_standard": sent_s,
                    "layer": layer_idx,
                    "layer_name": "embedding" if is_embedding else f"encoder_{layer_idx}",
                    "word_similarity": word_sim,
                    "sentence_similarity": sent_sim,
                    "attention_available": has_attentions and not is_embedding,
                    "intra_verlan": None if is_embedding else metric_at(attn_breakdown_v["intra"] if attn_breakdown_v else None, attn_idx),
                    "context_verlan": None if is_embedding else metric_at(attn_breakdown_v["context"] if attn_breakdown_v else None, attn_idx),
                    "self_verlan": None if is_embedding else metric_at(attn_breakdown_v["self"] if attn_breakdown_v else None, attn_idx),
                    "incoming_verlan": None if is_embedding else metric_at(incoming_v, attn_idx),
                    "intra_standard": None if is_embedding else metric_at(attn_breakdown_s["intra"] if attn_breakdown_s else None, attn_idx),
                    "context_standard": None if is_embedding else metric_at(attn_breakdown_s["context"] if attn_breakdown_s else None, attn_idx),
                    "self_standard": None if is_embedding else metric_at(attn_breakdown_s["self"] if attn_breakdown_s else None, attn_idx),
                    "incoming_standard": None if is_embedding else metric_at(incoming_s, attn_idx),
                    "target_subwords_verlan": len(word_tokens_v),
                    "target_subwords_standard": len(word_tokens_s),
                    "target_tokens_verlan": json_cell(word_tokens_v),
                    "target_tokens_standard": json_cell(word_tokens_s),
                    "token_span_verlan": json_cell(span_v),
                    "token_span_standard": json_cell(span_s),
                }
            )

    results_df = pd.DataFrame(layer_rows)
    tokenization_df = pd.DataFrame(tokenization_rows)
    failures_df = pd.DataFrame(
        failures,
        columns=["verlan", "standard", "template", "error"],
    )

    results_df.to_csv(output_dir / "experiment_results.csv", index=False)
    tokenization_df.to_csv(output_dir / "tokenization_results.csv", index=False)
    failures_df.to_csv(output_dir / "failures.csv", index=False)

    metadata = {
        "model_name": args.model_name,
        "device": str(device),
        "pairs_csv": args.pairs_csv,
        "templates_csv": args.templates_csv,
        "num_pairs": int(len(pairs_df)),
        "num_templates": int(len(templates_df)),
        "num_sentence_jobs": int(len(jobs_df)),
        "num_completed_jobs": int(len(tokenization_df) // 2),
        "num_failed_jobs": int(len(failures_df)),
        "attention_available_for_all_jobs": bool(
            len(results_df) > 0 and results_df.loc[results_df["layer"] > 0, "attention_available"].fillna(False).all()
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved experiment rows to {output_dir / 'experiment_results.csv'}")
    print(f"Saved tokenization rows to {output_dir / 'tokenization_results.csv'}")
    if len(failures_df) > 0:
        print(f"Saved {len(failures_df)} failures to {output_dir / 'failures.csv'}")


if __name__ == "__main__":
    main()
